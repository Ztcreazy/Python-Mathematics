import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
"""
The pickle module implements binary protocols for serializing 
and de-serializing a Python object structure. “Pickling” is the process 
whereby a Python object hierarchy is converted into a byte stream, 
and “unpickling” is the inverse operation, whereby a byte stream (from a binary file 
or bytes-like object) is converted back into an object hierarchy. 
Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1] 
or “flattening”; however, to avoid confusion, the terms used here are “pickling” 
and “unpickling”.
"""

"""
The Generator provides access to a wide range of distributions, 
and served as a replacement for RandomState. The main difference between the two is 
that Generator relies on an additional BitGenerator to manage state 
and generate the random bits, which are then transformed into random values 
from useful distributions. The default BitGenerator used by Generator is PCG64. 
The BitGenerator can be changed by passing an instantized BitGenerator to Generator.
"""

"""
PCG-64 is a 128-bit implementation of O’Neill’s permutation congruential generator 
([1], [2]). PCG-64 has a period of 2^128 and supports advancing an arbitrary number of steps 
as well as 2^127 streams.
"""

import os

def Q_Learning(episodes, is_training=True, render=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 30)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 30)    # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # 20*20*3
    else:
        f = open('ReinforcementLearning/RL01/mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    Learning_rate_a = 0.9 
    Discount_factor_g = 0.9 

    epsilon = 1 # 100%
    epsilon_decay_rate = 2/episodes # 2 / 5000

    rng = np.random.default_rng()   

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0] 
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False        

        rewards=0

        while(not terminated and rewards>-1000):

            if is_training and rng.random() < epsilon:
                
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + \
                                              Learning_rate_a * (reward + 
                                              Discount_factor_g * 
                                              np.max(q[new_state_p, new_state_v,:]) - 
                                              q[state_p, state_v, action]
                                              )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[episode] = rewards

    env.close()

    # Save Q table to file
    if is_training:
        f = open('ReinforcementLearning/RL01/mountain_car.pkl','wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    
    folder_path = 'ReinforcementLearning/RL01'
    plt.plot(mean_rewards)
    # plt.savefig(f'mountain_car.png')
    plt.savefig(os.path.join(folder_path, f'mountain_car.png'))

if __name__ == '__main__':
    
    # Q_Learning(5000, is_training=True, render=False)

    Q_Learning(10, is_training=False, render=True)