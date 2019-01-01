class _Getch:
	def __call__(self):
		#Each open file is given a unique file descriptor
		#In Unix, the three file descriptors are 0, 1, and 2
		#sys.stdin.fileno() is 0
		#sys.stdout.fileno() is 1
		#sys.stderr.fileno() is 2
		#the following code below gets single input from keyboard
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(3)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch

inkey = _Getch()

#MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


#Key Mapping
arrow_keys = {
	'\x1b[A': UP,
	'\x1b[B': DOWN,
	'\x1b[C': RIGHT,
	'\x1b[D': LEFT
}

import gym
from gym.envs.registration import register
import sys, tty, termios

register(
	id = 'FrozenLake-v3',
	entry_point = 'gym.envs.toy_text: FrozenLakeEnv',
	kwargs = {'map_name': '4x4', 'is_slippery': False}
)

#make the environment
env = gym.make('FrozenLake-v3')
env.render()


while True:
	key = inkey()
	if key not in arrow_keys.keys():
		print("Game aborted!")
		break
	action = arrow_keys[key]
	state, reward, done, info = env.step(action)
	env.render()
	print("State: ", state, "Action", action, "Reward: ", reward, "Info ", info)
	
	if done:
		print("Finished with reward", reward)
		break




