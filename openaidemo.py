import gym
import universe
import random


#reinforcement learning step
def determine_turn(turn, observation,j,total_sum,prev_total_sum,reward):
	#for every 15 iterations, sum the total observations
	#if lower than 0, change the direction

	if (j >= 15):

		if (total_sum % j == 0): #sameness
			turn = True
		else:
			turn = False

		#reset vars
		prev_total_sum = total_sum
		total_sum = 0
		j = 0

	else:
		turn = False

	if(observation != None):
		#increment counter and reward sum
		j+=1
		total_sum+=reward

	return (turn,j,total_sum,prev_total_sum)




def main():

	#init environment
	env = gym.make('flashgames.CoasterRacer-v0')
	env.configure(remotes=1)
	observation_n = env.reset()


	#init variables
	n = 0 #num iterations
	j = 0 

	# sum observations
	total_sum = 0
	prev_total_sum = 0

	turn = False

	#define actinos

	left = [('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowLeft', True),('KeyEvent', 'ArrowRight', False)]
	right = [('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', True)]
	forward = [('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', False)]

	#Main logic
	while True:
		##one iteration
		n+=1 

		#check if turn is needed
		if (n>1):

			#check if received input
			if (observation_n[0] != None):
				#store the reward in the previous score
				prev_score = reward_n[0]

				#should we turn
				if (turn):
					#where to turn
					event = random.choice([left,right])

					#perform an action
					action_n = [event for ob in observation_n]

					#set turn to false
					turn = False

		elif(~turn): #go straight
			action_n = [forward for ob in observation_n]

		if (observation_n[0] != None):

			turn,j,total_sum,prev_total_sum = determine_turn(turn, observation_n[0],j,total_sum,prev_total_sum,reward_n[0])

		#save variable
		observation_n,reward_n,done_n,info = env.step(action_n)

		env.render()

if __name__ == '__main__':
	main();
