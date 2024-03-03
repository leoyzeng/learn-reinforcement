import gym
from model import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':

    # environment
    env = gym.make('SpaceInvaders-v0')

    # create agent
    brain = Agent(gamma=0.95, epsilon=1.0, alpha=0.003, maxMemorySize=5000, replace=None)
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0: no action, 1: fire, 2: move right, 3: move left, 4: move right fire, 5: move left fire
            action = env.action_space.sample()
            observation_, reward, done, truncated, info = env.step(action)

            # heavy punishment for losing
            if done and info['lives'] == 0:
                reward = -100

            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200, 30:125], axis=2))
            # observation_ is successor state

    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size = 32 # make this smaller if too slow

    # play 50 games
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()

        frames = [np.sum(observation[15:200, 30:125], axis=2)]
        score = 0
        lastAction = 0

        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = lastAction

            observation_, reward, done, truncated, info = env.step(action) # take the action
            score += reward
            frames.append(np.sum(observation[15:200, 30:125], axis=2))

            # penalty for loosing
            if done and info['lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward, np.mean(observation_[15:200, 30:125], axis=2))

            observation = observation_
            brain.learn(batch_size)
            lastAction = action
            env.render()

        scores.append(score)
        print('score: ', score)

        x = [i + 1 for i in range(numGames)]
        fileName = 'test' + str(numGames) + '.png'
        plotLearning(x, scores, epsHistory, fileName)






