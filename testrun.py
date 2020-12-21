import catinous.CatsinomModelGramCache as catsmodel

# hparams={'continous': True,
#          'force_misclassified': True,
#          'datasetfile': 'catsinom_combined_dataset.csv',
#          'base_model': 'batch_lr_base_train_1_2d20289ac9.pt',
#          'val_check_interval': 30,
#          'cachemaximum': 512,
#          'run_postfix': 'test1'}
#
# model, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams, show_progress=True)

hparams={'continous':False,
         'datasetfile': 'catsinom_lr_dataset.csv',
         'noncontinous_train_splits': ['base_train'],
         'noncontinous_steps': 3000}
_, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams)

hparams={'continous': True,
         'use_cache': False,
         'datasetfile': 'catsinom_combined_hrlowshift_dataset.csv',
         'base_model': basemodel_lr,
         'EWC': True,
         'EWC_dataset': 'catsinom_lr_dataset.csv',
         'EWC_lambda': 1,
         'EWC_bn_off': True,
         'val_check_interval': 100}

# hparams={'continous': True,
#          'force_misclassified': True,
#          'datasetfile': 'catsinom_combined_hrlowshift_dataset.csv',
#          'base_model': basemodel_lr,
#          'val_check_interval': 99,
#          'cachemaximum': 64}

model, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams, show_progress=True)

# model = CatsinomModelGramCache(hparams=hparams, device=torch.device('cuda'))
# logger = utils.pllogger(model.hparams)
# trainer = Trainer(gpus=1, max_epochs=1, early_stop_callback=False, logger=logger, val_check_interval=model.hparams.val_check_interval, show_progress_bar=True, checkpoint_callback=False)
# trainer.fit(model)