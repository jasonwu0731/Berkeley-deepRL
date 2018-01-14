#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import os
import pandas as pd

from data_utils import *

def run_single_task(args, envname):
    env = gym.make(envname)

    if os.path.exists("experts/"+envname+".meta"):
        f = open("experts/"+envname+".meta", 'rb')
        data = pickle.load(f)
    else:
        data = generate_expert_data(envname, args.max_timesteps, "experts/"+envname+".pkl", args.num_rollouts)

    print(data["observations"].shape, data["actions"].shape)
    print(env.observation_space, env.action_space)


    grid = [1,2,5,10,15,20,25,30] if args.grid else [args.nb_epoch]
    grid_checkpoint = {}
    for idx in range(len(grid)):
        model = train_model(data, env, args, grid[idx])
        save = True if idx==len(grid)-1 else False
        imit_data = test_model(model, args, env, envname, save=save)
        grid_checkpoint[grid[idx]] = imit_data['returns']
    
    pickle_name = 'experts/{}-rewards.p'.format(envname)
    pickle.dump(grid_checkpoint, open(pickle_name, 'wb'))

def train_model(data, env, args, nb_epochs):
    from sklearn.utils import shuffle
    from keras.callbacks import EarlyStopping

    model = build_model(args, data, env)

    x, y = shuffle(data['observations'], data['actions'].reshape(-1, env.action_space.shape[0]))
    
    cb = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', verbose=2)] if args.es else []
    
    model.fit(x, y, validation_split=0.1, batch_size=args.bsz, nb_epoch=nb_epochs, callbacks=cb)

    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('expert_policy_file', type=str)
    #parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--es', action='store_true')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()

    TASK_LIST = [
        #"Ant-v1",
        #"HalfCheetah-v1",
        "Hopper-v1",
        #"Humanoid-v1",
        #"Reacher-v1",
        #"Walker2d-v1"
    ]

    for task in TASK_LIST:
        run_single_task(args, task)

    for task in TASK_LIST:
        data = pickle.load(open("experts/"+task+'.meta', 'rb'))
        imit_data = pickle.load(open("experts/"+task+'.imit.meta', 'rb'))
        compare_data(data, imit_data, task)


if __name__ == '__main__':
    main()
