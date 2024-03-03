# https://www.youtube.com/watch?v=ELE2_Mftqoc

import torch as T
import  torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


### network to play space invader

# first part of network process the input images
class DeepQNetwork(nn.Module):

    # ALPHA is learning rate
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        # 3 convolution layers to extract info from image
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # 2 fully connected layers to determin output
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6) # output is size 6, since there are 6 actions in space invader

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    # forward pass
    def forward(self, observation): # observation is sequence of frames
        observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95) # any number of frames, 185x95 size image
        observation = F.relu(self.conv1(observation)) # pass through CNN
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128*19*8) # flatten
        observation = F.relu(self.fc1(observation))

        actions = self.fc2(observation) # there is 1 set of actions per frame

        return actions

class Agent(object):

    # gamma is discount factor (reward now is worth more than reward in future)
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, replace=10000, actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = [] # memory is stored in a list
        self.memCntr = 0 # total memory stored
        self.replace_target_cnt = replace # how often to replace target network
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    # store state, action, reward in memory
    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize: # if there is memory available, save to memory
            self.memory.append([state, action, reward, state_])
        else: # overwrite earlier memory if out of space
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    # pass in sequence of frames, output action
    def chooseAction(self, observation):
        rand = np.random.random() # random number for epsilon greedy action selection
        actions = self.Q_eval.forward(observation) # do forward pass through neural network
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1].item()) # take the greedy action
        else:
            action = np.random.choice(self.actionSpace) # take a random action
        self.steps += 1
        return action

    # pass in batch size
    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad() # zero out gradients since we are doing batch optimization
        if (self.replace_target_cnt is not None) and (self.learn_step_counter % self.replace_target_cnt == 0):
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # calculate start index of memory subsampling
        if self.memCntr + batch_size < self.memSize: # make sure we don't go outside the memory
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memCntr-batch_size-1)))

        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device) # current set of states
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        # max action for next state
        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        # calculate rewards
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        # target and predicted values used to update loss funciton
        Qtarget = Qpred
        Qtarget[:, maxA] = rewards + self.GAMMA*T.max(Qnext[1]) # max action for next successor state

        # we want agent to slowly converge on purely greedy strategy
        if self.steps > 500: # wait 500 steps first
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4 # converge towards greedy
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device) # loss function
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1









if __name__ == '__main__':

    pass
