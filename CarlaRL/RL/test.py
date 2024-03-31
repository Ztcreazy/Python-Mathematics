from stable_baselines3 import PPO
from carenv import CarEnv

models_dir = "CarlaRL/RL/models/1711881785"

env = CarEnv()
env.reset()

model_path = f"{models_dir}/2000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # env.render()
        print(reward)