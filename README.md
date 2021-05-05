# Dynamic memory to alleviate catastrophic forgetting in continual learning with medical imaging

In submission at _Nature Communications_.

**1. Requirements**



**2. Data set**

The data set used can be downloaded from the Multi-Centre, Multi-Vendor & Multi-Disease 
Cardiac Image Segmentation Challenge (M&Ms), https://www.ub.edu/mnms/.

After downloading it is necessary to preprocess the data by running:

`python data_prep/data_prep_cardiac.py <download_path> <dataset_path>`

where `<download_path>` is the directory where the M&Ms dataset was downloaded to, and `<dataset_path>` is where the preprocessed data is stored.

**3. Training**

To run the training the config files in `training_configs/` are used. Please modify output and input directories as needed.

`python run_training.py --config training_configs/cardiac_base.yml`

`python run_training.py --config training_configs/cardiac_dynamicmemory.yml`

To run the entire training as reported in the paper use:
`python run_training.py --paper`

_Please note: Running the entire training runs multiple training runs for each setting! Therefore, a long runtime is expected._

**4. Results**
