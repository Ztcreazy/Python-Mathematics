# import gym
import gymnasium as gym
# from gym import spaces
from gymnasium import spaces
from stable_baselines3 import A2C

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLanderContinuous-v2", render_mode = "human")
env.reset()

if isinstance(env.action_space, spaces.Discrete):
    print("env action space dicrete")
    print("env action space n: ", env.action_space.n)
    print("env action space sample: ", env.action_space.sample())

elif isinstance(env.action_space, spaces.Box):
    print("env action space continuous")
    print("env action space shape: ", env.action_space.shape)
    print("env action space sample: ", env.action_space.sample())

# print("env observation space shape: ", env.observation_space.shape)
# print("env observation space sample: ", env.observation_space.sample())

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

episodes = 10

for i in episodes:
    obs = env.reset()
    terminated = False
    while not terminated:
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        # print("reward: ", reward)

env.close()