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
'batch_size':64,
'learning_rate': 'scheduled',
'lr_boundaries':[int(120e3),int(140e4),int(160e4)],
'lr_values':[1e-3,1e-4,1e-5,1e-6],
'layer':np.array([400, 400, 400]),
'maxsteps':False,
'env':'Humanoid-v1',
'pretrained_model': '/'.join(str.split(current_dir, '/')[:-1]) + '/sched1/modeldata/model95002',
'niter_dagger':100,
'min_additional_datapoints':2000,
'train_iter_per_dagger': 1000
}
