import gym
import numpy as np
import matplotlib.pyplot as plt 
from gym.envs.registration import register
import random as pr

def rargmax(vector):
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return pr.choice(indices)

register(
	id = 'FrozenLake-v3', 
	entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
	kwargs = {'map_name': '4x4', 'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

#initialize table with all zeros by np.zeros()
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Set learning parameters. Iterate 2000 times to get Q table
num_episodes = 2000

#create lists to contain total rewards and steps per episode

rList = []
for i in range(num_episodes):
	#reset environment and get first new observation
	state = env.reset()
	rAll = 0
	done = False
	
	#Algorithm to get the Q-table
	#if done becomes true or the state reaches to Hole, skip the loop
	
	while not done:
		#we will choose the action which has maximum reward, but if every action returns the same amount of reward, we will choose the action randomly. Therefore, in this case we use "rargmax()"
		action = rargmax(Q[state, :])
		
		#Get new state and reward from environment
		#"env.step(action)" function executes the action and gets state, reward, done information
		new_state, reward, done,_ = env.step(action)

		Q[state, action] = reward + np.max(Q[new_state, :]) # ":" means every case
		rAll += reward
		state = new_state

		rList.append(rAll)

print("Success rate:" + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()
