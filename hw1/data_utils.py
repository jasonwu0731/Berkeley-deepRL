import pickle
import numpy as np
import load_policy
import tf_util
import tensorflow as tf
import pandas as pd

def generate_expert_data(envname, max_timesteps, expert_policy_file, num_rollouts, save=True):
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(expert_policy_file)
        print('loaded and built')

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                #print("action", action)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1 
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

    #print('returns', returns)
    #print('mean return', np.mean(returns))
    #print('std of return', np.std(returns))

    expert_data = {
                'observations': np.array(observations),
                'actions': np.array(actions),
                'returns': np.array(returns)
    }
    
    if save:
        f = open("experts/"+envname+'.meta','wb')
        pickle.dump(expert_data, f)

    return expert_data

def data_table_stats(data):
    mean = data['returns'].mean()
    std = data['returns'].std()
    return pd.Series({
        'mean reward': mean,
        'std reward': std,
    })

def compare_data(data, imit_data, envname):
    df = pd.DataFrame({
        'expert': data_table_stats(data),
        'imitation': data_table_stats(imit_data)
    })

    print("Analyzing experiment", envname)
    print(df)

def build_model(args, data, env):
    from keras.models import Sequential
    from keras.layers import Dense, Lambda
    from keras.optimizers import Adam

    mean, std = np.mean(data['observations'],axis=0), np.std(data["observations"], axis=0) + 1e-5

    model = Sequential()
    model.add(Lambda(lambda x: (x-mean)/std, batch_input_shape=(None,env.observation_space.shape[0] )))
    model.add(Dense(args.hidden, activation="tanh"))
    model.add(Dense(args.hidden, activation="tanh"))
    model.add(Dense(env.action_space.shape[0]))

    opt = Adam(lr=args.lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mse"])
    return model

def test_model(model, args, env, envname, save=True):
    max_steps = env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)

    imitation_data = {'observations': np.array(observations),
                    'actions': np.array(actions),
                    'returns': np.array(returns)
    }
    if save:
        pickle.dump(imitation_data, open("experts/"+envname+'.imit.meta','wb'))
    return imitation_data

def run_expert_on_observations(observations, expert_policy_file):
    policy_fn = load_policy.load_policy(expert_policy_file)
    with tf.Session():
        tf_util.initialize()
        actions = []
        for obs in observations:
            action = policy_fn(obs[None,:])
            actions.append(action)
    return np.array(actions)