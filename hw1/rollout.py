#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import numpy as np
import gym
import load_policy
import tensorflow as tf
import tf_util


def rollout(envname, max_timesteps=None, render=True, num_rollouts=20):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy("experts/" + envname + ".pkl")
    print('loaded and built')

    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    with tf.Session():
        tf_util.initialize()

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('rollout iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
