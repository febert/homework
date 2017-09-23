import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import importlib.machinery
import sys
import numpy as np
from datetime import datetime
import pickle
import pdb
import random

import matplotlib.pyplot as plt

import load_policy

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_bool('test', False, 'whether to test the model in mujoco')
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')

flags.DEFINE_string('render', False, 'whether to render the environment')

flags.DEFINE_bool('aggregate', False, 'whether to test the model in mujoco')

flags.DEFINE_bool('loop_over_rollouts', False, 'loop_over_num_rollouts')


# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 200
# How often to run a batch through the validation model.
VAL_INTERVAL = 400
# How often to save a model checkpoint
SAVE_INTERVAL = 5000


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Generic_Model(object):
    def __init__(self, conf):
        self.m = BC_model
        self.conf = conf

        import gym
        self.env = gym.make(self.conf['env'])
        self.obs_dim = int(self.env.observation_space.high.shape[0])
        self.action_dim = int(self.env.action_space.high.shape[0])

        self.inferred_action = None

        if not FLAGS.test:
            self.global_step = tf.Variable(0, trainable=False)
            if conf['learning_rate'] == 'scheduled':
                print('using scheduled learning rate')
                self.lr = tf.train.piecewise_constant(self.global_step, conf['lr_boundaries'], conf['lr_values'])
            else:
                self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

            with open(conf['train_data'], 'rb') as f:
                data = pickle.loads(f.read())

            for k in data.keys():
                data[k] = np.squeeze(data[k])

            train_val_split = .9
            total_num_examples = data['observations'].shape[0]

            self.train_data = {}
            self.val_data = {}
            for k in data.keys():
                self.train_data[k] = data[k][:int(total_num_examples * train_val_split)]
                self.val_data[k] = data[k][np.int(total_num_examples * train_val_split):]

            self.train_summaries = []
            self.val_summaries = []

            assert self.obs_dim == data['observations'].shape[1]
            assert self.action_dim == data['actions'].shape[1]

            self.true_action = tf.placeholder(tf.float32, name='gtruth_action',
                                              shape=(self.conf['batch_size'], self.action_dim))

        else:
            self.conf['batch_size'] = 1

        self.obs = tf.placeholder(tf.float32, name='observations',
                                  shape=(self.conf['batch_size'], self.obs_dim))

    def init_sess(self):
        # pdb.set_trace()
        print('Initializing Session...')
        # Make saver.

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # remove all states from group of variables which shall be saved and restored:
        self.saver = tf.train.Saver(vars, max_to_keep=0)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # Make training session.
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        self.summary_writer = tf.summary.FileWriter(self.conf['output_dir'], graph=self.sess.graph, flush_secs=10)

        self.sess.run(tf.global_variables_initializer())

        itr_0 = 0
        if self.conf['pretrained_model'] != '':
            self.saver.restore(self.sess, self.conf['pretrained_model'])

        if FLAGS.test:
            self.saver.restore(self.sess, self.conf['pretrained_model'])
            print('finished loading weights.')

        return itr_0

    def build_optimizer(self):
        print('building optimizer')
        self.loss = mean_squared_error(self.true_action, self.inferred_action)

        self.train_summaries.append(tf.summary.scalar('loss', self.loss))
        self.val_summaries.append(tf.summary.scalar('val_loss', self.loss))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

        self.train_summ_op = tf.summary.merge(self.train_summaries)
        self.val_summ_op = tf.summary.merge(self.val_summaries)


    def train(self, init_sess = True, num_iterations=None):
        """
        :param init_sess: build optimizer and initialize variables
        :param num_iterations:
        :return:
        """

        if init_sess:
            self.build_optimizer()
            itr_0 = self.init_sess()
        else: itr_0 = 0

        num_train_ex = self.train_data['observations'].shape[0]
        num_val_ex = self.val_data['observations'].shape[0]

        t_iter = []

        if num_iterations is None:
            num_iterations = self.conf['num_iterations']

        for itr in range(itr_0, num_iterations, 1):
            t_startiter = datetime.now()
            # Generate new batch of data_files.

            train_data_ind = np.random.choice(num_train_ex, self.conf['batch_size'])

            feed_dict = {
                self.obs: np.squeeze(self.train_data['observations'][train_data_ind]),
                self.true_action: np.squeeze(self.train_data['actions'][train_data_ind])
            }
            cost, _, summary_str = self.sess.run([self.loss, self.train_op, self.train_summ_op],
                                                 feed_dict)

            if (itr) % 500 == 0:
                tf.logging.info('iter'+str(self.sess.run(self.global_step)) + ' ' + str(cost))

            if (itr) % VAL_INTERVAL == 2:
                val_data_ind = np.random.choice(num_val_ex, self.conf['batch_size'])
                # Run through validation set.
                feed_dict = {
                             self.obs: self.val_data['observations'][val_data_ind],
                             self.true_action: self.val_data['actions'][val_data_ind]
                }
                [val_summary_str] = self.sess.run([self.val_summ_op], feed_dict)
                self.summary_writer.add_summary(val_summary_str, self.sess.run(self.global_step))

            if (itr) % SAVE_INTERVAL == 2:
                tf.logging.info('Saving model to' + self.conf['output_dir'])
                self.saver.save(self.sess, self.conf['output_dir'] + '/model' + str(self.sess.run(self.global_step)))

            t_iter.append((datetime.now() - t_startiter).seconds * 1e6 + (datetime.now() - t_startiter).microseconds)


            if (itr) % SUMMARY_INTERVAL:
                self.summary_writer.add_summary(summary_str, self.sess.run(self.global_step))

    def policy_fn(self, obs):
        obs = obs.reshape(1,self.obs_dim)
        obs = np.repeat(obs, self.conf['batch_size'], 0)

        feed_dict = {self.obs: obs}
        inferred_action = self.sess.run([self.inferred_action], feed_dict)
        inferred_action = inferred_action[0][0]
        return inferred_action

    def run_trajectories(self, num_rollouts=20, init_sess=True):
        """
        :param num_rollouts:
        :param init_sess: if True will initialize weights randomly and load weights
        :return:
        """

        if init_sess:
            self.init_sess()

        max_steps = self.conf['maxsteps'] or self.env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        expert_actions = []

        for i in range(num_rollouts):
            print('num_rollout', i)
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = self.policy_fn(obs)
                observations.append(obs)
                actions.append(action)

                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if FLAGS.render:
                    self.env.render()
                if steps % 200 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        return np.mean(returns), np.std(returns)


    def aggregate_dataset(self):
        print('starting dagger')

        self.init_sess()

        self.agg_traindataset = {}
        for k in self.train_data.keys():
            self.agg_traindataset[k] = np.split(self.train_data[k], self.train_data[k].shape[0], 0)

        self.agg_valdataset = {}
        for k in self.train_data.keys():
            self.agg_valdataset[k] = np.split(self.val_data[k], self.val_data[k].shape[0], 0)

        max_steps = self.conf['maxsteps'] or self.env.spec.timestep_limit

        self.conf['num_iterations'] = self.conf['niter_dagger'] * self.conf['train_iter_per_dagger']

        expert_policy_fn = load_policy.load_policy('experts/{}.pkl'.format(self.conf['env']))
        print('expert policy loaded and built')

        mean_return_list = []
        std_dev_list = []

        for d in range(self.conf['niter_dagger']):
            print('starting running dagger-iteration ', d)

            n_added = 0
            returns = []

            while n_added < self.conf['min_additional_datapoints']:
                obs = self.env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = self.policy_fn(obs)

                    if FLAGS.aggregate:
                        expert_action = expert_policy_fn(obs[None,:])

                    obs, r, done, _ = self.env.step(action)
                    totalr += r
                    steps += 1
                    if FLAGS.render:
                        self.env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))


                    if (steps % 20) == 0:
                        self.agg_valdataset['observations'].append(obs)
                        self.agg_valdataset['actions'].append(expert_action)
                    else:
                        self.agg_traindataset['observations'].append(obs)
                        self.agg_traindataset['actions'].append(expert_action)
                        n_added += 1

                    if steps >= max_steps:
                        break

                returns.append(totalr)
                print('return: ', totalr)
                print('length of dataset: ', len(self.agg_traindataset['observations']))

            mean_return_list.append(np.mean(returns))
            std_dev_list.append(np.std(returns))

            if d == 0:
                init_sess = True
            else:
                init_sess = False
            self.train(init_sess=init_sess, num_iterations=self.conf['train_iter_per_dagger'])

        pickle.dump([mean_return_list, std_dev_list],
                    open(self.conf['output_dir'] + '/returns.pkl', 'wb'))

        self.plot_dagger(mean_return_list, std_dev_list)


    def loop_over_numrollouts(self):

        with open(self.conf['train_data'], 'rb') as f:
            fulldata = pickle.loads(f.read())

        for k in fulldata.keys():
            fulldata[k] = np.squeeze(fulldata[k])

        train_val_split = .9
        total_num_examples = fulldata['observations'].shape[0]

        self.train_data = {}
        self.val_data = {}

        steps_per_traj = 1000
        sizes = np.array(range(1,20,2))*steps_per_traj
        # sizes = np.array([100000])

        mean_return_list = []
        std_dev_list = []

        for size in sizes:
            print('using size: {} steps'.format(size))
            data_subset = {}
            for k in fulldata.keys():
                data_subset[k] = fulldata[k][:size]

            self.train_data = {}
            self.val_data = {}
            for k in data_subset.keys():
                self.train_data[k] = data_subset[k][:int(size * train_val_split)]
                self.val_data[k] = data_subset[k][np.int(size * train_val_split):]

            self.train()
            mean_return, std_dev = self.run_trajectories(num_rollouts=10, init_sess=False)
            mean_return_list.append(mean_return)
            std_dev_list.append(std_dev)

        pickle.dump([sizes, mean_return_list, std_dev_list],
                    open( self.conf['output_dir'] + '/returns.pkl','wb'))


        self.plot_train_data(sizes, mean_return_list, std_dev_list)


    def plot_train_data(self, sizes =None, mean_return_list=None, std_dev_list=None):
        if type(sizes) is not np.ndarray:
            with open('/home/frederik/Documents/courses/DeepRL/homework/hw1/experiments/Hopper-v1/loop_over_numrollouts/modeldata/returns.pkl', 'rb') as f:
                sizes, mean_return_list, std_dev_list = pickle.loads(f.read())

        sizes = sizes.astype(np.float32) / 1000
        plt.figure()
        ax = plt.gca()
        ax.errorbar(list(sizes), mean_return_list, yerr=std_dev_list)
        ax.set_title('Performance of BC for Hopper-v1 over number of observed trajectories')
        plt.xlabel('number of rollouts (1k steps each) used for training')
        plt.ylabel('average return/ std. deviation')

        plt.show()


    def plot_dagger(self, mean_return_list= None, std_dev_list= None):
        if mean_return_list is None:
            # file = '/home/frederik/Documents/courses/DeepRL/homework/hw1/experiments/Humanoid-v1/dagger2/modeldata/returns.pkl'
            file = '/home/frederik/Documents/courses/DeepRL/homework/hw1/experiments/Humanoid-v1/dagger/modeldata/returns.pkl'
            with open(file, 'rb') as f:
                mean_return_list, std_dev_list = pickle.loads(f.read())

        plt.figure()
        ax = plt.gca()
        h1 = ax.errorbar(range(len(mean_return_list)), mean_return_list, yerr=std_dev_list, label='Dagger')

        h2, = ax.plot([0, len(mean_return_list)-1], [10414, 10414], color='k', linestyle='--', label='Expert')
        h3, = ax.plot([0, len(mean_return_list) - 1], [1261, 1261], color='r', linestyle='--', label='BC')

        plt.legend(handles = [h1, h2, h3])

        ax.set_title('Performance for Humanoid-v1 over number of dagger iterations')
        plt.xlabel('dagger iterations')
        plt.ylabel('average return/ std. deviation')

        plt.savefig(self.conf['output_dir'] + '/output.png')
        plt.show()

class BC_model(Generic_Model):
    def __init__(self, conf):
        Generic_Model.__init__(self, conf)

        layer_sizes = conf['layer']
        nlayer = len(layer_sizes)

        out1 = slim.layers.fully_connected(
            self.obs,
            int(layer_sizes[0]))

        out2 = slim.layers.fully_connected(
            out1,
            int(layer_sizes[1]))

        out3 = slim.layers.fully_connected(
            out2,
            int(layer_sizes[2]))

        self.inferred_action = slim.layers.fully_connected(
            out3,
            int(self.action_dim),
            activation_fn=None)


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    conf_file = FLAGS.hyper

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = importlib.machinery.SourceFileLoader('conf', conf_file).load_module()

    conf = hyperparams.configuration

    m = BC_model(conf)

    # m.plot_dagger()
    # sys.exit()

    if FLAGS.test:
        m.run_trajectories(num_rollouts=20)
    elif FLAGS.loop_over_rollouts:
        m.loop_over_numrollouts()
    elif FLAGS.aggregate:
        m.aggregate_dataset()
    else:
        m.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()



