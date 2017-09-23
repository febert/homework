import numpy as np
import tensorflow as tf
import gym
import logz
import plot
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#


def gaussian_log_prob(x, mean, logstd):
    std = tf.exp(logstd)
    logprob = -0.5 * tf.reduce_sum(tf.square(tf.divide(x - mean, std)) - 0.5 * tf.log(2 * np.pi) - tf.log(std), axis=1)
    return logprob


def gaussian_sample(mean, std):
    return tf.contrib.distributions.MultivariateNormalDiag(mean, tf.exp(std)).sample()


def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
        ):

    with tf.variable_scope(scope):
        print("building {} with {} hidden layers and {} units each".format(scope, n_layers, size))

        output = None
        for l in range(n_layers):  # before n_layers -1!!!
            print("making layer{}".format(l))
            if l == 0:
                input = input_placeholder
            else:
                input = output
            output = tf.layers.dense(input, size, activation=activation)

        output = tf.layers.dense(output, output_size, activation=output_activation)

        return output


#ignasi's mlp
# def normc_initializer(std=1.0):
#     """
#     Initialize array with normalized columns
#     """
#
#     def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
#         out = np.random.randn(*shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         return tf.constant(out)
#
#     return _initializer
# def fclayer(x, size, name, weight_init=normc_initializer(std=0.1)):
#     """
#     fully connected layer
#     """
#     w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
#     b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)
#     return tf.matmul(x, w) + b
# def build_mlp(
#         obs,
#         act_dim,
#         scope='policy',
#         n_layers=2,
#         size=64,
#         hidden_nonlinearity=tf.nn.tanh,
#         output_nonlinearity=None, ):
#
#     hidden_layers = n_layers * (size,)
#     l_hid = obs
#     hidden_nonlinearity = tf.identity if hidden_nonlinearity is None else hidden_nonlinearity
#     output_nonlinearity = tf.identity if output_nonlinearity is None else output_nonlinearity
#
#     with tf.variable_scope(scope):
#         # self._layers = [l_hid]
#         for idx, num_units in enumerate(hidden_layers):
#             l_hid = fclayer(l_hid, num_units, 'hidden%d' % idx)
#             l_hid = hidden_nonlinearity(l_hid)
#         # self._layers.append(l_hid)
#         l_out = output_nonlinearity(fclayer(l_hid, act_dim, 'output'))
#     # self._layers.append(self.l_out)
#     return l_out

def pathlength(path):
    return len(path["reward"])


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=64,
             init_var=1e-2,
             baseline_learning_rate = 5e-3
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.

    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(shape=[None], name="advantage", dtype=tf.float32)

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = build_mlp(sy_ob_no, ac_dim, 'policy_network')

        sy_prob_n_pol = tf.nn.softmax(sy_logits_na)
        sy_logprob_n_pol = tf.log(tf.nn.softmax(sy_logits_na))
        sy_sampled_ac = tf.multinomial(sy_logprob_n_pol, 1)

        # dist = tf.contrib.distributions.Multinomial(total_count=1., logits=sy_logprob_n_pol)
        # sy_logprob_n = tf.log(dist.prob(tf.cast(sy_ac_na, tf.float32)))

        #only works for nactios = 2
        assert int(sy_prob_n_pol.get_shape()[1]) == 2
        sy_ac_na = tf.cast(sy_ac_na, tf.float32)
        sy_logprob_n = tf.log(tf.pow(sy_prob_n_pol[:,0], 1-sy_ac_na) * tf.pow(sy_prob_n_pol[:,1], sy_ac_na))
    else:

        sy_mean = build_mlp(sy_ob_no, ac_dim, 'policy_network', n_layers=n_layers, size= size)


        sy_logstd = tf.Variable(initial_value=0 * tf.ones(shape=(ac_dim,), dtype=tf.float32),
                                name='policy_std')  # logstd should just be a trainable variable, not a network output.
        sy_sampled_ac = gaussian_sample(sy_mean, sy_logstd)
        sy_logprob_n = gaussian_log_prob(sy_ac_na, sy_mean,
                                         sy_logstd)  # Hint: Use the log probability under a multivariate gaussian.

        #mine
        # sy_logstd = tf.Variable(np.ones(ac_dim)*init_var, name='logstd' , dtype=tf.float32, trainable=True) # logstd should just be a trainable variable, not a network output.  #0.1
        #
        # sigma = tf.diag(tf.exp(sy_logstd))
        # sy_sampled_ac = sy_mean + tf.reshape(tf.matmul(sigma, tf.expand_dims(tf.random_normal(shape=[ac_dim]), 1)), [1, ac_dim])
        #
        # inv_sigma = tf.matrix_inverse(sigma)
        #
        # sy_logprob_n = -0.5 * tf.reduce_sum(tf.multiply(sy_mean - sy_ac_na, tf.matmul(sy_mean - sy_ac_na, inv_sigma)), axis=1) - 0.5* tf.log(tf.matrix_determinant(2*np.pi*sigma))

        # the contrib way
        # dist = tf.contrib.distributions.MultivariateNormalDiag(sy_mean, tf.exp(sy_logstd))
        # sy_sampled_ac = dist.sample([int(sy_adv_n.get_shape()[0])])
        # sy_logprob_n = dist.log_pdf(sy_ac_na)  # Hint: Use the log probability under a multivariate gaussian.

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    # Loss function that we'll differentiate to get the policy gradient.
    loss = -tf.reduce_mean(sy_logprob_n * sy_adv_n)

    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1, 
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 

        sy_target_n = tf.placeholder(shape=[None], name="target", dtype=tf.float32)
        b_loss = tf.reduce_mean(tf.square(baseline_prediction - sy_target_n))

        baseline_update_op = tf.train.AdamOptimizer(baseline_learning_rate).minimize(b_loss)

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101


    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            next_obs, obs, acs, rewards =[], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob.copy())
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = np.squeeze(ac[0])
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                next_obs.append(ob.copy())
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs),
                    "next_observation": np.array(next_obs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])

        ac_na = np.concatenate([path["action"] for path in paths])
        rew_na = np.concatenate([path["reward"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        if not reward_to_go:
            Q_t = []
            for path in paths:
                path_l = len(path["reward"])
                ret = 0
                for t in range(path_l):
                    ret += path["reward"][t]*np.power(gamma,t)

                ret = np.repeat(ret, path_l)
                Q_t.append(ret)
            q_n = np.concatenate(Q_t,axis=0)
        else:
            Q_t = []
            # Q_t_slow = []
            # # the slow way:
            # for path in paths:
            #     q_t = 0
            #     path_l  = len(path["reward"])
            #     for t in range(path_l):
            #
            #         ret = 0
            #         for tprime in range(t,path_l):
            #             ret += path["reward"][tprime] * np.power(gamma, tprime-t)
            #
            #         Q_t_slow.append(ret)
            # q_n_slow = np.array(Q_t_slow)

            for path in paths:
                q_t = 0
                path_l  = len(path["reward"])

                ret_tp1 = 0
                dis_ret = []
                for t in range(path_l-1, -1, -1):
                    ret_tp1 = path["reward"][t] + gamma*ret_tp1
                    dis_ret.append(ret_tp1)  # This has to be done later on?

                dis_ret = np.flip(np.array(dis_ret), axis=0)

                Q_t.append(dis_ret)

            q_n = np.concatenate(Q_t)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        use_td_bl = False

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            feed_dict = {sy_ob_no: ob_no}
            b_n = sess.run(baseline_prediction, feed_dict=feed_dict)

            if not use_td_bl:
                b_n = b_n*np.std(q_n) + np.mean(q_n)
                # b_n = (b_n - np.mean(b_n)) / (np.std(b_n)+1e-6) * np.std(q_n) + np.mean(q_n)

            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 

            adv_n = (adv_n - np.mean(adv_n))/(np.std(adv_n) + 1e-6)

            pass

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            if use_td_bl:
                b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: next_ob_no})
                b_n = (b_n - np.mean(q_n)) / (np.std(q_n) + 1e-6)

                target_n = rew_na + gamma * b_n
                target_n = (target_n - np.mean(q_n)) / (np.std(q_n) + 1e-6)
            else:
                target_n = (q_n - np.mean(q_n)) / (np.std(q_n) + 1e-6)

            feed_dict = {sy_target_n: target_n,
                         sy_ob_no: ob_no}

            _, b_loss_npy = sess.run([baseline_update_op, b_loss], feed_dict=feed_dict)

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.
        if not discrete and ac_dim == 1:
            ac_na = np.expand_dims(ac_na, 1)

        if reward_to_go:
            adv_scale = np.concatenate([gamma ** np.array(list(range(len(path["reward"])))) for path in paths])
            adv_n *= adv_scale

        feed_dict = {sy_adv_n: adv_n,
                     sy_ob_no: ob_no,
                     sy_ac_na: ac_na}

        loss_before_up = sess.run(loss, feed_dict=feed_dict)
        sess.run(update_op, feed_dict=feed_dict)
        loss_after_up = sess.run(loss, feed_dict=feed_dict)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)

        logz.log_tabular("advantage_var", np.var(adv_n))
        logz.log_tabular("advantage_mean", np.mean(adv_n))
        logz.log_tabular("loss_before_up", loss_before_up)
        logz.log_tabular("loss_after_up", loss_after_up)

        if nn_baseline:
            logz.log_tabular("b_loss", b_loss_npy)

        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--init_var', '-ivar', type=float, default=-1)
    parser.add_argument('--baseline_learning_rate', '-blrf', type=float, default=5e-3)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                init_var=args.init_var,
                baseline_learning_rate=args.baseline_learning_rate
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
