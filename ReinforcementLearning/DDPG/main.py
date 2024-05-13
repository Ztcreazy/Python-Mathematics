import torch
import gym
import numpy as np

from ddpg import DDPGAgent

env = gym.make("Pendulum-v1")

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# hyperparameters
MAX_EPISODES = 200
MAX_STEPS = 200
BATCH_SIZE = 64

for episode in range(MAX_EPISODES):
    
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):

        # [-1, 1]
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM)
        next_state, reward, done, _, _ = env.step( 2 *action )

        agent.memory.push(state, action, reward, next_state, done)

        episode_reward += reward
        state = next_state

        if len(agent.memory) > BATCH_SIZE:

            agent.update(BATCH_SIZE)

        if done:

            break

    print(f"Episode: {episode+1}, Reward: {episode_reward}")

env.close()

torch.save(agent.actor.state_dict(), 'C:/Users/14404/OneDrive/Desktop/PythonMathematics/ReinforcementLearning/DDPG/actor.pth')
torch.save(agent.critic.state_dict(), 'C:/Users/14404/OneDrive/Desktop/PythonMathematics/ReinforcementLearning/DDPG/critic.pth')
