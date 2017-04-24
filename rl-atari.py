import gym
import numpy as np
import cv2
import sys
import copy
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K

#sources
#dqn-atari - https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
#dqn - https://keon.io/deep-q-learning/
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


#Specific Processing for atari games
class AtariProcessor(Processor):
	def __init__(self,input_size):
		self.input_size = input_size

    def process_observation(self, observation):
        assert observation.ndim == 3 
        #image processing goes here
        proc_observation= cvImgProcChain(observation).resize(self.input_size).gray().show('processed').process()
        return proc_observation

    def process_state_batch(self, batch):
        # batch processing for compressino
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


#Convolutional Neural Networks
def build_model(out_actions,in_shape):
	K.set_image_dim_ordering('tf')

	model = Sequential()
	model.add(Permute((2,3,1),input_shape=in_shape))
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(out_actions))
	model.add(Activation('linear'))
	print(model.summary())
	#plot_model(model)
	return model
	
			
##chain proc method class
class cvImgProcChain:
	def __init__(self,img):
		self.img = img
		self.sigmaX = 1
	#here goes image processing
	def blur(self):
		self.img = cv2.GaussianBlur(self.img, self.img.size, self.sigmaX)
		return self
	def gray(self):
		self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		return self
	def resize(self,size):
		self.img = cv2.resize(self.img, size)
		return self
	
	def to_cv2_color(self):
		cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		return self
	
	def show(self,winname):
		cv2.imshow(winname, self.img)
		cv2.waitKey(1)
		return self

	def stop(self):
		cv2.waitKey(0)
		return self

	def process(self):
		return self.img

#compression size
INPUT_SIZE = (84, 84)
WINDOW_LENGTH = 4


def main():

	#args for options
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['train', 'test'], default='train')
	parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v3')
	parser.add_argument('--weights', type=str, default=None)
	args = parser.parse_args()

	#init environment
	env = gym.make(args.env_name)
	#env.configure(remotes=1)

	#randomize for new experiences! 
	np.random.seed(123)
	env.seed(123)

	#number of possible actions/outputs
	num_actions = env.action_space.n

	#input shape after compression
	input_shape = (WINDOW_LENGTH,) + INPUT_SIZE

	#build model
	model = build_model(num_actions,input_shape)

	#initialize processor
	processor = AtariProcessor(INPUT_SIZE)

	#initialize memory for batch processing
	memory = SequentialMemory(limit=1000000, 
		window_length=4)

	#greed policy for training based on random actions
	policy = LinearAnnealedPolicy(
		EpsGreedyQPolicy(), attr='eps', 
		value_max=1., value_min=.1, value_test=.05,nb_steps=1000000)

	#Agent construct and compile
	dqn = DQNAgent(model=model,nb_actions=num_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
	dqn.compile(Adam(lr=.00025), metrics=['mae'])

	#if begin training e train our dqn agent
	if args.mode == 'train':
		#training weights callbacks allow for interrupts
	    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
	    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
	    log_filename = 'dqn_{}_log.json'.format(args.env_name)
	    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
	    callbacks += [FileLogger(log_filename, interval=100)]
	    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

	    # save weights
	    dqn.save_weights(weights_filename, overwrite=True)

	    # evaluate algorithm for 10 episodes
	    dqn.test(env, nb_episodes=10, visualize=False)

	#else run our best weights
	elif args.mode == 'test':
	    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
	    if args.weights:
	        weights_filename = args.weights
	    dqn.load_weights(weights_filename)
	    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
	main();

