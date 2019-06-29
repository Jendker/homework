import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from rollout import rollout


def load_network(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy(obs_bo):
        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        def apply_nonlin(x):
            if nonlin_type == 'lrelu':
                return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b)

        # Output layer
        W, b = read_layer(policy_params['out'])
        output_bo = tf.matmul(curr_activations_bd, W) + b
        return output_bo

    obs_bo = tf.placeholder(tf.float32, [None, None])
    a_ba = build_policy(obs_bo)
    # policy_fn = tf_util.function([obs_bo], a_ba)
    return obs_bo, a_ba


def create_model(input_size, output_size):
    # create inputs
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

    # create variables
    W0 = tf.get_variable(name='W0', shape=[input_size, 256], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name='W1', shape=[256, 64], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[64, output_size], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[256], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[64], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[output_size], initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    # create computation graph
    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred


env = 'Humanoid-v2'
iters = 10

with tf.Session() as sess:
    tf_util.initialize()

    example_data_file = open('expert_data/' + env + '.pkl', 'rb')
    example_data = pickle.loads(example_data_file.read())
    inputs = np.squeeze(example_data['observations'])
    outputs = np.squeeze(example_data['actions'])
    input_ph, output_ph, output_pred = create_model(input_size=inputs.shape[1], output_size=outputs.shape[1])
    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # initialize variables
    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()

    batch_size = 128

    for i in range(0, iters):
        print('-------------- Total iteration', i, '--------------')
        rollout(env, render=False, num_rollouts=40, max_timesteps=400)
        with open('expert_data/' + env + '.pkl', 'rb') as f:
            data = pickle.loads(f.read())
            inputs = np.squeeze(data['observations'])
            outputs = np.squeeze(data['actions'])

            # run training
            for training_step in range(2500):
                # get a random subset of the training data
                indices = np.random.randint(low=0, high=inputs.shape[0], size=batch_size)
                input_batch = inputs[indices, :]
                output_batch = outputs[indices, :]

                # run the optimizer and get the mse
                _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})
                if training_step % 1000 == 0:
                    print('{0:04d} mse: {1:.4f}'.format(training_step, mse_run))
            saver.save(sess, '/tmp/model.ckpt')

    saver.save(sess, 'learned_policies/' + env + '.pkl')

