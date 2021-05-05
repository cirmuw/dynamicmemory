import argparse
import logging
import os
from pprint import pprint

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

from dataset.BatchDataset import *
from dataset.ContinuousDataset import *
sys.path.append('../')
import utils as dmutils
from dynamicmemory.DynamicMemory import DynamicMemory, MemoryItem


class DynamicMemoryModel(pl.LightningModule):

    def __init__(self, hparams={}, modeldir = None, device=torch.device('cpu'), verbose=False, training=True):
        super(DynamicMemoryModel, self).__init__()
        self.hparams = argparse.Namespace(**hparams)
        self.to(device)
        self.modeldir = modeldir

        # load model according to hparams
        self.model, self.stylemodel, self.gramlayers = dmutils.load_model_stylemodel(self.hparams.task)
        self.stylemodel.to(device)
        self.seperatestyle = True

        self.forcemisclassified = True if 'force_misclassified' in self.hparams else False
        self.pseudo_detection = True if 'pseudodomain_detection' in self.hparams else False
        self.useewc = True if 'EWC' in self.hparams else False

        if self.hparams.task == 'lidc':
            self.collate_fn = dmutils.collate_fn
            self.TaskDatasetBatch = LIDCBatch
            self.TaskDatasetContinuous = LIDCContinuous
            self.task_validation_step = self.lidc_validation_step
            self.task_validation_end = self.lidc_validation_end
            self.get_task_loss = self.get_lidc_loss
            self.get_forcemetrics = self.get_forcemetrics_lidc
            self.force_element = lambda m: m > self.hparams.misclass_threshold
        elif self.hparams.task == 'cardiac':
            self.collate_fn = None
            self.TaskDatasetBatch = CardiacBatch
            self.TaskDatasetContinuous = CardiacContinuous
            self.task_validation_step = self.cardiac_validation_step
            self.task_validation_end = self.cardiac_validation_end
            self.loss = nn.CrossEntropyLoss()
            self.get_task_loss = self.get_cardiac_loss
            self.get_forcemetrics = self.get_forcemetrics_cardiac
            self.force_element = lambda m: m < self.hparams.misclass_threshold

        if self.useewc:
            if self.hparams.task == 'cardiac':
                self.ewcloss = dmutils.CrossEntropyWithEWCLoss(torch.tensor([self.hparams.EWC_lambda]))
                self.ewc = self.init_ewc()
                self.get_task_loss = self.get_ewc_loss
            else:
                raise NotImplementedError('EWC is only implemented for cardiac segmentation')

        if not self.hparams.base_model is None:
            state_dict = torch.load(os.path.join(modeldir, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('model.', '')] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        if training:
            # Initilize checkpoints to calculate BWT, FWT after training
            self.scanner_checkpoints = dict()
            self.scanner_checkpoints[self.hparams.order[0]] = True
            for scanner in self.hparams.order[1:]:
                self.scanner_checkpoints[scanner] = False

        # Initialize memory and hooks
        if self.hparams.use_memory and self.hparams.continuous and training:
            self.init_memory_and_gramhooks()
        else:
            self.hparams.use_memory = False

        if verbose:
            pprint(vars(self.hparams))

    def init_ewc(self):
        dl = DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                              split=['base'],
                                              res=self.hparams.order[0]),
                        batch_size=8, num_workers=8, drop_last=True)
        self.model.cuda()
        ewc = dmutils.EWC(self.model, dl)

        return ewc

    def init_memory_and_gramhooks(self):
        if self.hparams.gram_weights is None:
            self.hparams.gram_weights = [1] * len(self.gramlayers)

        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        if self.pseudo_detection:
            base_if, base_transformer = self.get_base_domainclf()
        else:
            base_if = None
            base_transformer = None

        self.trainingsmemory = DynamicMemory(memorymaximum=self.hparams.memorymaximum,
                                             balance_memory=self.hparams.balance_memory,
                                             gram_weights=self.hparams.gram_weights,
                                             base_if=base_if,
                                             base_transformer=base_transformer,
                                             seed=self.hparams.seed)

        logging.info('Gram hooks and cache initialized. Memory size: %i' % self.hparams.memorymaximum)

    def get_base_domainclf(self):
        self.freeze()
        dl = DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                              split=['base'],
                                              res=self.hparams.order[0]),
                        batch_size=8, num_workers=8, drop_last=True)

        base_grams = []
        for j, batch in enumerate(dl):
            self.grammatrices = []
            torch.cuda.empty_cache()

            images, targets, scanner, filepath = batch

            x = images.to(self.device)
            _ = self.forward(x)

            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu().detach().numpy().flatten() for bg in self.grammatrices]
                base_grams.append(np.hstack(grammatrix))

            self.grammatrices = []

        transformer = SparseRandomProjection(random_state=self.hparams.seed, n_components=30)
        transformer.fit(base_grams)
        trans_initelements = transformer.transform(base_grams)

        clf = IsolationForest(n_estimators=10, random_state=self.hparams.seed).fit(trans_initelements)
        self.unfreeze()
        return clf, transformer

    def gram_hook(self, m, input, output):
        self.grammatrices.append(dmutils.gram_matrix(input[0]))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, scanner, filepath = batch
        self.grammatrices = []

        if self.hparams.continuous:
            # save checkpoint at scanner shift
            self.checkpoint_scanner_shift(scanner)

        # train with memory
        if self.hparams.use_memory and self.hparams.continuous:
            self.freeze()
            # add new batch to memory
            self.grammatrices = []
            if type(x) is list:
                imgsx = torch.stack(x)
                _ = self.stylemodel(imgsx.float())
            else:
                _ = self.stylemodel(x)

            # to force misclassified a forward pass with the frozen model is needed
            if self.forcemisclassified:
                if type(x) is list:
                    imgsx = torch.stack(x)
                    y_hat = self.forward(imgsx.float())
                else:
                    y_hat = self.forward(x.float())

                forcemetrics = self.get_forcemetrics(y, y_hat)

            forcedelements = []
            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in self.grammatrices]
                target = y[i]
                if type(target) == torch.Tensor:
                    det_target = target.detach().cpu()
                else:
                    det_target = {}
                    for k, v in target.items():
                        det_target[k] = v.detach().cpu()

                mi = MemoryItem(img.detach().cpu(), det_target, filepath[i], scanner[i], grammatrix)
                self.trainingsmemory.insert_element(mi)

                if self.forcemisclassified and self.force_element(forcemetrics[i]):
                    forcedelements.append(mi)

            # pseudo domains are used to balance the memory
            if self.pseudo_detection:
                self.trainingsmemory.check_outlier_memory(self)
                self.trainingsmemory.counter_outlier_memory()
                for i in range(len(self.trainingsmemory.isoforests)):
                    domainitems = self.trainingsmemory.get_domainitems(i)
                    counts = dict()
                    for o in self.hparams.order:
                        counts[o] = 0
                    for mi in domainitems:
                        counts[mi.scanner] += 1

            self.unfreeze()

            x, y = self.trainingsmemory.get_training_batch(self.hparams.training_batch_size, forceditems=forcedelements)

        loss = self.get_task_loss(x, y)
        self.grammatrices = []
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.hparams.continuous:
            return torch.optim.Adam(self.parameters(), lr=0.00001)
        else:
            return torch.optim.Adam(self.parameters(), lr=0.0001)

    def checkpoint_scanner_shift(self, scanner):
        newshift = False
        for s in scanner:
            if s != self.hparams.order[0] and not self.scanner_checkpoints[s]:
                newshift = True
                shift_scanner = s
        if newshift:
            exp_name = dmutils.get_expname(self.hparams)
            weights_path = self.modeldir + exp_name + '_shift_' + shift_scanner + '.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.scanner_checkpoints[shift_scanner] = True

    def train_dataloader(self):
        if self.hparams.continuous:
            return DataLoader(self.TaskDatasetContinuous(self.hparams.datasetfile,
                                                         transition_phase_after=self.hparams.transition_phase_after,
                                                         seed=self.hparams.seed,
                                                         order=self.hparams.order),
                              batch_size=self.hparams.batch_size, num_workers=8, drop_last=True,
                              collate_fn=self.collate_fn)
        else:
            return DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                                    iterations=self.hparams.noncontinuous_steps,
                                                    batch_size=self.hparams.batch_size,
                                                    split=self.hparams.noncontinuous_train_splits,
                                                    res=self.hparams.scanner),
                              batch_size=self.hparams.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                                split='val', res=self.hparams.order),
                          batch_size=4,
                          num_workers=2,
                          collate_fn=self.collate_fn)

    def validation_step(self, batch, batch_idx):
        self.grammatrices = []
        x, y, scanner, _ = batch
        return self.task_validation_step(x, y, scanner)

    def validation_epoch_end(self, validation_step_outputs):
        self.task_validation_end(validation_step_outputs)

    def get_forcemetrics_cardiac(self, y, y_hat):
        forcemetrics = []
        y_hat_flat = torch.argmax(y_hat['out'], dim=1).detach().cpu().numpy()
        y_det = y.detach().cpu().numpy()
        for i, m in enumerate(y):
            forcemetrics.append(dmutils.dice(y_det[i], y_hat_flat[i], classi=1))

        return forcemetrics

    def get_forcemetrics_lidc(self, y, y_hat):
        forcemetrics = []
        out_boxes = [dmutils.filter_boxes_area(y_hat[i]['boxes'].cpu().detach().numpy(),
                                               y_hat[i]['scores'].cpu().detach().numpy()) for i in
                     range(len(y_hat))]
        boxes_np = [b[0] for b in out_boxes]
        scores_np = [b[1] for b in out_boxes]
        for i, box_np in enumerate(boxes_np):
            fb, fs = dmutils.correct_boxes(box_np, scores_np[i])
            fneg = dmutils.get_false_negatives(fb, fs, y[i]['boxes'])
            forcemetrics.append(fneg)
        return forcemetrics

    def get_lidc_loss(self, x, y):
        x = list(i.to(self.device) for i in x)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        loss_dict = self.forward_lidc(x, targets)
        loss = sum(l for l in loss_dict.values())

        return loss

    def forward_lidc(self, x, y):
        return self.model(x, y)

    def get_cardiac_loss(self, x, y):
        x = x.to(self.device)
        if type(y) is list:
            y = torch.stack(y).to(self.device)
        y_hat = self.forward(x.float())
        loss = self.loss(y_hat['out'], y)

        return loss

    def get_ewc_loss(self, x, y):
        x = x.to(self.device)

        y_hat = self.forward(x.float())
        loss = self.ewcloss(y_hat['out'], y, self.ewc.penalty(self.model))

        return loss

    def cardiac_validation_step(self, x, y, scanner):
        scanners = []
        dice_1 = []
        dice_2 = []
        dice_3 = []

        y_hat = self.forward(x)['out']
        y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for i, m in enumerate(y):
            scanners.append(scanner[i])
            dice_1.append(dmutils.dice(y[i], y_hat_flat[i], classi=1))
            dice_2.append(dmutils.dice(y[i], y_hat_flat[i], classi=2))
            dice_3.append(dmutils.dice(y[i], y_hat_flat[i], classi=3))
        return {'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3}

    def lidc_validation_step(self, images, targets, scanner):
        images = list(image.to(self.device) for image in images)

        out = self.model(images)

        out_boxes = [
            dmutils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(), out[i]['scores'].cpu().detach().numpy())
            for i in range(len(out))]

        boxes_np = [b[0] for b in out_boxes]
        scores_np = [b[1] for b in out_boxes]

        final_boxes = []
        final_scores = []
        for i, box_np in enumerate(boxes_np):
            fb, fs = dmutils.correct_boxes(box_np, scores_np[i])
            final_boxes.append(fb)
            final_scores.append(fs)

        gt = []
        for t in targets:
            gt.append(t['boxes'])

        return {'final_boxes': final_boxes, 'final_scores': final_scores, 'gt': gt, 'scanner': scanner}

    def lidc_validation_end(self, validation_step_outputs):
        iou_thres = 0.2

        overall_true_pos = dict()
        overall_false_pos = dict()
        overall_false_neg = dict()
        overall_boxes_count = dict()
        recalls = dict()
        precision = dict()

        for scanner in self.hparams.order:
            overall_true_pos[scanner] = dict()
            overall_false_pos[scanner] = dict()
            overall_false_neg[scanner] = dict()
            overall_boxes_count[scanner] = dict()
            recalls[scanner] = []
            precision[scanner] = []
            for k in np.arange(0.0, 1.01, 0.05):
                overall_true_pos[scanner][k] = 0
                overall_false_pos[scanner][k] = 0
                overall_false_neg[scanner][k] = 0
                overall_boxes_count[scanner][k] = 0

        for out in validation_step_outputs:
            final_boxes = out['final_boxes']
            final_scores = out['final_scores']
            gt = out['gt']
            scanner = out['scanner']

            for j, fb in enumerate(final_boxes):
                s = scanner[j]
                g = gt[j]
                fs = final_scores[j]

                for k in np.arange(0.0, 1.01, 0.05):
                    false_positives = 0
                    false_negatives = 0
                    true_positives = 0
                    detected = [False] * len(g)
                    boxes_count = 0
                    if len(fb) > 0:
                        for i, b in enumerate(fb):
                            if fs[i] > k:
                                boxes_count += 1
                                det_gt = False
                                for m, singleg in enumerate(g):
                                    if dmutils.bb_intersection_over_union(singleg, b) > iou_thres:
                                        detected[m] = True
                                        det_gt = True
                                if not det_gt:
                                    false_positives += 1
                    for d in detected:
                        if d:
                            true_positives += 1
                        else:
                            false_negatives += 1
                    overall_true_pos[s][k] += true_positives
                    overall_false_pos[s][k] += false_positives
                    overall_false_neg[s][k] += false_negatives
                    overall_boxes_count[s][k] += boxes_count

        aps = dict()

        for scanner in self.hparams.order:
            for k in np.arange(0.0, 1.01, 0.05):
                if (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]) == 0:
                    recalls[scanner].append(0.0)
                else:
                    recalls[scanner].append(
                        overall_true_pos[scanner][k] / (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]))
                if (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]) == 0:
                    precision[scanner].append(0.0)
                else:
                    precision[scanner].append(
                        overall_true_pos[scanner][k] / (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]))

            prec = np.array(precision[scanner])
            rec = np.array(recalls[scanner])
            ap = []
            for t in np.arange(0.0, 1.01, 0.1):
                prec_arr = prec[rec > t]
                if len(prec_arr) == 0:
                    ap.append(0.0)
                else:
                    ap.append(prec_arr.max())
            aps[scanner] = np.array(ap).mean()

            self.log(f'val_ap_{scanner}', aps[scanner])

    def cardiac_validation_end(self, validation_step_outputs):
        scanners = []
        dice_1 = []
        dice_2 = []
        dice_3 = []

        for p in validation_step_outputs:
            scanners.extend(p['scanner'])
            dice_1.extend(p['dice_1'])
            dice_2.extend(p['dice_2'])
            dice_3.extend(p['dice_3'])

        df_dice = pd.DataFrame({'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3})
        df_mean = df_dice.groupby('scanner').mean()
        for s in df_mean.index:
            self.log(f'val_dice1_{s}', df_mean['dice_1'][s])
            self.log(f'val_dice2_{s}', df_mean['dice_2'][s])
            self.log(f'val_dice3_{s}', df_mean['dice_3'][s])


def trained_model(hparams, settings, training=True):
    df_memory = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    settings = argparse.Namespace(**settings)
    os.makedirs(settings.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(settings.TRAINED_MEMORY_DIR, exist_ok=True)
    os.makedirs(settings.RESULT_DIR, exist_ok=True)

    model = DynamicMemoryModel(hparams=hparams, device=device, training=training)
    exp_name = dmutils.get_expname(hparams)
    print('expname', exp_name)
    weights_path = cached_path(hparams, settings.TRAINED_MODELS_DIR)

    if not os.path.exists(weights_path):
        if training:
            logger = dmutils.pllogger(hparams, settings.LOGGING_DIR)
            trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                              val_check_interval=model.hparams.val_check_interval,
                              checkpoint_callback=False, progress_bar_refresh_rate=0)
            trainer.fit(model)
            model.freeze()
            torch.save(model.state_dict(), weights_path)
            if model.hparams.continuous and model.hparams.use_memory:
                dmutils.save_cache_to_csv(model.trainingsmemory.memorylist,
                                          settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            model = None
    else:
        print('Read: ' + cached_path(hparams))
        model.load_state_dict(torch.load(cached_path(hparams), map_location=device))
        model.freeze()

    if model.hparams.continuous and model.hparams.use_memory:
        if os.path.exists(settings.TRAINED_MEMORY_DIR + exp_name + '.csv'):
            df_memory = pd.read_csv(settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            df_memory = None

    # always get the last version
    if os.path.exists(settings.LOGGING_DIR + exp_name):
        max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
        logs = pd.read_csv(settings.LOGGING_DIR + exp_name + '/version_{}/metrics.csv'.format(max_version))
    else:
        logs = None

    return model, logs, df_memory, exp_name + '.pt'


def is_cached(hparams, trained_dir):
    exp_name = dmutils.get_expname(hparams)
    return os.path.exists(trained_dir + exp_name + '.pt')


def cached_path(hparams, trained_dir):
    exp_name = dmutils.get_expname(hparams)
    return trained_dir + exp_name + '.pt'
