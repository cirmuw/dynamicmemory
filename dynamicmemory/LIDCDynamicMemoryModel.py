from dataset.BatchDataset import *
from dataset.ContinuousDataset import *
sys.path.append('../')
import utils as dmutils
from dynamicmemory.DynamicMemoryModel import DynamicMemoryModel


class LIDCDynamicMemoryModel(DynamicMemoryModel):
    def __init__(self, hparams={}, modeldir = None, device=torch.device('cpu'), training=True):
        super(DynamicMemoryModel, self).__init__()
        self.init(hparams=hparams, modeldir=modeldir, device=device, training=training)

        self.collate_fn = dmutils.collate_fn
        self.TaskDatasetBatch = LIDCBatch
        self.TaskDatasetContinuous = LIDCContinuous

    def force_element(self, m):
        return m > self.hparams.misclass_threshold

    def get_forcemetrics(self, y, y_hat):
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

    def get_task_loss(self, x, y):
        x = list(i.to(self.device) for i in x)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        loss_dict = self.forward_lidc(x, targets)
        loss = sum(l for l in loss_dict.values())

        return loss

    def forward_lidc(self, x, y):
        return self.model(x, y)

    def validation_step(self, batch, batch_idx):
        self.grammatrices = []
        images, targets, scanner, _ = batch
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

    def validation_epoch_end(self, validation_step_outputs):
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
