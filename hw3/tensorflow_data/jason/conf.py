import os
import pdb
current_dir = os.path.dirname(os.path.realpath(__file__))

# local output directory
OUT_DIR = current_dir + '/modeldata'

import jason_dqn

configuration = {
'experiment_name': 'rndaction_var10',
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'double_q':'',
'jason_model':""
}
