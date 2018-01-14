import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

from data_utils import *

"""
Dataset aggregation.
Steps:
    1) train cloned policy Pi(u_t, o_t) from expert data
    2) run Pi(u_t, o_t) to get data set D_pi = {o_1, ... , o_M}
    3) Ask human to label D_pi with actions
    4) Aggregate the dataset
"""

def run_dagger(args, env_name):
    env = gym.make(env_name)
    actions_dim = env.action_space.shape[0]

    training_opts = {
        "validation_split": 0.1, 
        "batch_size": args.bsz,
        "nb_epoch": args.nb_epoch,
        "verbose": 2
    }

    data = generate_expert_data(env_name, None, "experts/"+env_name+".pkl", 1, save=False)

    x = data['observations']
    y = data['actions'].reshape(-1, actions_dim)

    model = build_model(args, data, env)

    stats = {}
    rewards = {}

    for i in range(30):
        print("DAgger iter:", i)
        # 1) train cloning policy from expert data
        x, y = shuffle(x, y)
        model.fit(x, y, **training_opts)

        # 2) run cloning policy to get new data
        data = test_model(model, args, env, env_name, save=False)

        new_x = data['observations']
        stats[i] = data_table_stats(data)
        rewards[i] = data['returns']

        # 3) ask expert to label D_pi with actions
        new_y = run_expert_on_observations(new_x, "experts/"+env_name+".pkl")

        # 4) Aggregate the dataset
        x = np.append(x, new_x, axis=0)
        y = np.append(y, new_y.reshape(-1, actions_dim), axis=0)

    df = pd.DataFrame(stats).T
    df.index.name = 'iterations'
    df.to_csv('experts/{}-DAgger.csv'.format(env_name))
    pickle_name = 'experts/{}-DAgger-rewards.p'.format(env_name)
    pickle.dump(rewards, open(pickle_name, 'wb'))

def show_results(env_name):
    
    pickle_name = 'experts/{}-DAgger-rewards.p'.format(env_name)
    DAgger_returns = pickle.load(open(pickle_name, 'rb'))
    df = pd.DataFrame(DAgger_returns)

    expert_data = pickle.load(open('experts/{}.meta'.format(env_name), 'rb'))['returns']
    expert_data = np.array([expert_data for _ in range(30)]).T

    # need to run $python run_BC.py --grid
    if os.path.exists('experts/{}-rewards.p'.format(env_name)):
        imit_data = pickle.load(open('experts/{}-rewards.p'.format(env_name), 'rb'))
        vanilla_data = pd.DataFrame(imit_data)

    if os.path.exists('experts/{}-rewards.p'.format(env_name)):
        sns.tsplot(time=vanilla_data.columns, data=vanilla_data.values, color='purple', linestyle=':')
    sns.tsplot(time=df.columns, data=df.values, color='blue', linestyle='-')
    sns.tsplot(data=expert_data, color='red', linestyle='--')
    #sns.tsplot(data=imit_data, color='purple', linestyle=':')

    plt.ylabel("Mean reward")
    plt.xlabel("Number of epochs (vanilla cloning), number of iterations (DAgger)")

    plt.title("{} - Comparison of DAgger and vanilla behavioral cloning policies".format(env_name))

    import matplotlib.patches as mpatches
    if os.path.exists('experts/{}-rewards.p'.format(env_name)):
        plt.legend(handles=[
            mpatches.Patch(color='purple', label='Vanilla cloning policy'),
            mpatches.Patch(color='blue', label='DAgger policy'),
            mpatches.Patch(color='red', label='expert policy'),
        ], loc='lower right')

    plt.savefig('imgs/dagger-vanilla-comp-{}.png'.format(env_name))
    plt.close()

if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('expert_policy_file', type=str)
    #parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--bsz', type=int, default=64)
    #parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of expert roll outs')
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
        run_dagger(args, task)
        show_results(task)
