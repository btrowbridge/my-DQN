import gym
import universe
import numpy as np
import cv2
import sys
import copy

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K

#sources
#atari - https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
#dqn - https://keon.io/deep-q-learning/

INPUT_SHAPE = (210, 160)
WINDOW_LENGTH = 3

class DQNAgent:

	def __init__(self,env):
		self.env = env
		self.memory = []
		self.gamma = 0.9  # decay rate
		self.epsilon = 1e-08  # exploration
		self.decay = 0.0
		self.learning_rate = 0.0001
		self.beta1 = 0.9
		self.beta2 =0.999
		self.input_shape = env.observation_space.shape
		self.num_actions = env.action_space.n
		self._build_model()

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			# The agent acts randomly
			return self.env.action_space.sample()
		# Predict the reward value based on the given state
		act_values = self.model.predict(state[None,:])
	  
		# Pick the action based on the predicted reward
		return np.argmax(act_values[0])

	def _build_model(self):
		K.set_image_dim_ordering('tf')

		model = Sequential()
		model.add(Permute((1,2,3),input_shape=self.input_shape))
		model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(self.num_actions))
		model.add(Activation('linear'))
		print(model.summary())
		model.compile(
			Adam(lr=self.learning_rate,beta_1=self.beta1,beta_2=self.beta2,epsilon=self.epsilon,decay=self.decay), 
			loss = 'mse')
		#plot_model(model)
		self.model = model
		

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay(self, batch_size):
		batches = min(batch_size, len(self.memory))
		batches = np.random.choice(len(self.memory), batches)
		for i in batches:
			state, action, reward, next_state, done = self.memory[i]
			target = reward
			if not done:
			  target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state[None,:])[0])
			target_f = self.model.predict(state[None,:])
			target_f[0][action] = target
			self.model.fit(state[None,:], target_f, nb_epoch=1, verbose=1)
#		if self.epsilon > self.epsilon_min:
#			self.epsilon *= self.epsilon_decay
			
##chain proc method class
class ImgProc:
	def __init__(self,img):
		self.img
		self.sigmaX = 1
	#here goes image processing
	def blur(self):
		return cv2.GaussianBlur(self.img, self.img.size, self.sigmaX)
		
	def gray(self):
		return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

	def to_cv2_color(self):
		return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

	def show(self,winname):
		cv2.imshow(winname, img)
		cv2.waitKey(1)
		return img

def main():
	#init environment
	env = gym.make('Breakout-v0')
	#env.configure(remotes=1)

	#build model
	agent = DQNAgent(env)

	episodes = sys.argv[1] if len(sys.argv) > 1 else 100
	

	done = False
	for e in range(episodes):
	#Main logic
		prev_state = env.reset()

		for time_t in range(5000):
			if(prev_state[0]!= None):
				action = agent.act(prev_state)
			else:
				action = env.action_space.sample()

			next_state,reward,done,_ = env.step(action)

			reward = -100 if done else reward

			agent.remember(prev_state, action, reward, next_state, done)

			prev_state = copy.deepcopy(next_state)

			env.render()

		agent.replay(32)

if __name__ == '__main__':
	main();

