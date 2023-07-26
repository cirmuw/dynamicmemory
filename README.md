# Dynamic memory to alleviate catastrophic forgetting in continual learning with medical imaging

Code corresponding to the paper Perkonigg, M., Hofmanninger, J., Herold, C.J. et al. Dynamic memory to alleviate catastrophic forgetting in continual learning with medical imaging. Nat Commun 12, 5678 (2021). https://doi.org/10.1038/s41467-021-25858-z

**1. Requirements**

All experiments are performed with PyTorch 1.7.1 using Python 3.6. The requirements are given in `requirements.txt`.

Estimated install time on a PC: 30-60 minutes.

Original experiments were performed on NVIDIA GPUs with Linux CentOS 7.

**2. Data set**

The data set used can be downloaded from the Multi-Centre, Multi-Vendor & Multi-Disease 
Cardiac Image Segmentation Challenge (M&Ms), https://www.ub.edu/mnms/.

After downloading it is necessary to preprocess the data by running:

```python data_prep/data_prep_cardiac.py <download_path> <dataset_path>```

where `<download_path>` is the directory where the M&Ms dataset was downloaded to, and `<dataset_path>` is where the preprocessed data is stored.

**3. Training**

To run the training the config files in `training_configs/` are used. Please modify output and input directories as needed.

`python run_training.py --config training_configs/cardiac_base.yml`

`python run_training.py --config training_configs/cardiac_dynamicmemory.yml`

To run the entire training as reported in the paper use:
`python run_training.py --paper`

_Please note: Running the entire training runs multiple training runs for each setting! Therefore, a long runtime is expected._

**4. Results**

For convenience examples of a full analysis for cardiac segmentation are given in `evaluation/cardiac_evaluation.ipynb`.
