import os
import pdb
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
TRAIN_DATA = '/'.join(str.split(current_dir, '/')[:-2]) + '/training_data/Hopper-v1/trainingdata_100rollouts.pkl'
TEST_DATA = '/'.join(str.split(current_dir, '/')[:-2]) + '/training_data/Hopper-v1/testdata_10rollouts.pkl'

# local output directory
OUT_DIR = current_dir + '/modeldata'

import numpy as np

configuration = {
'experiment_name': 'rndaction_var10',
'train_data': TRAIN_DATA,       # 'directory containing data.' ,
'test_data': TEST_DATA,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'batch_size':64,
'learning_rate': 0.001,
'layer':np.array([400, 400, 400])
}
