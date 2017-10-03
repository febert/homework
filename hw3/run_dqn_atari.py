import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


from dqn_utils import *
from atari_wrappers import *

import os
import importlib
import sys

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
flags.DEFINE_bool('nogpu', False ,'the value for CUDA_VISIBLE_DEVICES variable')

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(conf, env,
                session,
                num_timesteps):

    if 'jason_model' in conf:
        import jason_dqn as dqn
    else:
        import dqn

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        conf,
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main(args):
    if FLAGS.nogpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    conf_file = FLAGS.hyper

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = importlib.machinery.SourceFileLoader('conf', conf_file).load_module()

    conf = hyperparams.configuration

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(conf, env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
