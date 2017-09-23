import os
import pdb
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
TRAIN_DATA = '/'.join(str.split(current_dir, '/')[:-3]) + '/training_data/Humanoid-v1/trainingdata_100rollouts.pkl'

# local output directory
OUT_DIR = current_dir + '/modeldata'

import numpy as np

configuration = {
'experiment_name': 'rndaction_var10',
'train_data': TRAIN_DATA,       # 'directory containing data.' ,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 100000,   #'number of training iterations.' ,
'batch_size':64,
'learning_rate': 'scheduled',
'lr_boundaries':[int(1e4),int(2e4),int(5e4)],
'lr_values':[1e-3,1e-4,1e-5,1e-6],
'layer':np.array([400, 400, 400]),
'maxsteps':False,
'env':'Humanoid-v1',
'pretrained_model': current_dir + '/modeldata/model194006',
}
