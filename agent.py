import numpy as np
import gym
import tiles
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

from model import Q_Network
import torch
import torch.nn.functional as F

import sys


filename = "set_filename"

def print_and_log(string):
    global filename
    print (string)
    with open(filename, 'a') as f:
        f.write(str(string)+ "\n")

class BaseAgent:
    def __init__(self, env, epochs=4000):
        self.env = env
        self.epochs = epochs
        self.avg_timesteps = []

    def learn(self):
        ''' 

        '''
        self.totalNumberOfTimeSteps = 0
        for i in range(self.epochs):
            self.state = self.env.reset()
            # self.state = self.state - self.state
            self.t = 0
            while True:
                if i > 10000:
                    self.env.render()
                # import pdb
                # pdb.set_trace()
                action = self.getAction(self.state)
                new_state, reward, done, _ = self.env.step(action)
                # new_state -= self.state
                # if done:
                #     reward = -100
                self.update(self.state, action, reward,
                            new_state, self.t, done)
                self.state = new_state
                self.t += 1
                if done:
                    self.episodeEnded()
                    self.totalNumberOfTimeSteps += self.t
                    # print_and_log("Episode finished after {} timesteps".format(self.t+1))
                    break

    def getAction(self, state):
        '''

        '''
        return bool(state[2] > 0)

    def update(self, state, action, reward, new_state, timestep, terminal_state):
        '''

        '''
        pass

    def episodeEnded(self):
        pass

    def log(self):
        print_and_log(self.env)


class Buffer(list):
    def __init__(self, capacity):
        self.capacity = capacity
        self.curr_pos = 0
        super(Buffer, self).__init__()

    def append(self, x):
        if (len(self) >= self.capacity):
            self[self.curr_pos % self.capacity] = x
        else:
            super(Buffer, self).append(x)
        self.curr_pos += 1


class ExperienceReplayAgent(BaseAgent):
    def __init__(self,
                 env: gym.Env,
                 epochs=10100,
                 epsilon=0.1,
                 N=250,
                 M=64,
                 num_train_iters=50,
                 num_ep_update_value=600,
                 gamma=1.0,
                 layer_size=[64],
                 lr=1e-3):

        super(ExperienceReplayAgent, self).__init__(env, epochs=epochs)
        self.experience = []
        self.epsilon = epsilon
        self.num_episodes = 0

        self.num_ep_update_value = num_ep_update_value

        self.N = N
        self.M = M
        self.num_train_iters = num_train_iters

        self.lr = lr
        self.layer_size = layer_size

        self.Q_new = Q_Network(
            input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.n, layer_size=layer_size)
        self.Q_old = Q_Network(
            input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.n, layer_size=layer_size)

        self.Q_old.load_state_dict(self.Q_new.state_dict())
        self.Q_old.eval()
        self.optimizer = torch.optim.Adam(self.Q_new.parameters(), lr=lr)

        self.alpdecay = 1
        self.alpha = 0.2
        self.gamma = gamma
        
    def getAction(self, state):
        '''

        '''
        if self.num_episodes <= self.N:
            return self.env.action_space.sample()

        if np.random.binomial(1, self.epsilon) == 0:
            q_values = self.Q_new(torch.from_numpy(
                state).float().unsqueeze(0)).detach().numpy()
            return np.argmax(q_values)
        else:
            return self.env.action_space.sample()

    def update(self, state, action, reward, new_state, timestep, terminal_state):
        '''

        '''
        self.experience.append(
            (state, action, reward, new_state, terminal_state))

    def episodeEnded(self):
        '''

        '''
        self.num_episodes += 1
        if (self.num_episodes % self.N == 0):
            print_and_log("Updating Q")
            self.batch_update()

        if (self.num_episodes % self.num_ep_update_value == 0):
            self.Q_old.load_state_dict(self.Q_new.state_dict())

    def getQValue(self, state, action):
        return self.Q_new(state)[action]

    def train_replay(self, x):
        state, action, rewards, next_state, final = x
        # import pdb; pdb.set_trace()
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state[~final, :]).float()
        rewards = torch.from_numpy(rewards).float()

        q_values = self.Q_new(state)[range(state.shape[0]), action].squeeze()
        v_values = torch.zeros(state.shape[0])
        v_values[torch.Tensor(final.tolist()) == False] = self.Q_old(
            next_state).max(1)[0].detach()

        expected_q_values = v_values * self.gamma + rewards

        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        for p in self.Q_new.parameters():
            p.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        # print_and_log([np.sum(np.abs(p.grad.data.numpy()))
        #        for p in self.Q_new.parameters()])
        # self.Q_new.eval()

    def batch_update(self):
        self.avg_timesteps.append((self.totalNumberOfTimeSteps / self.N))
        print_and_log("Avg timesteps per episode: %f" % self.avg_timesteps[-1])
        print_and_log("Num eps: %d" % self.num_episodes)
        print_and_log("Experience length: %d" % len(self.experience))
        for i in range(self.num_train_iters):
            batch_index = np.random.randint(len(self.experience), size=self.M)
            exp = np.array(self.experience)[batch_index]

            batch = (np.array(exp[:, 0].tolist()), np.array(exp[:, 1].tolist()), np.array(
                exp[:, 2].tolist()), np.array(exp[:, 3].tolist()), np.array(exp[:, 4].tolist()))
            self.train_replay(batch)
        self.totalNumberOfTimeSteps = 0

    def log(self):
        super(ExperienceReplayAgent, self).log()

        print_and_log("epochs=%d" % self.epochs) 
        print_and_log("epsilon=%f" % self.epsilon) 
        print_and_log("N=%d" % self.N) 
        print_and_log("M=%d" % self.M) 
        print_and_log("num_train_iters=%d" % self.num_train_iters) 
        print_and_log("num_ep_update_value=%d" % self.num_ep_update_value) 
        print_and_log("gamma=%f" % self.gamma) 
        print_and_log("layer_size=%s" % str(self.layer_size)) 
        print_and_log("lr=%f" % self.lr)



class SlowDownExperienceReplayAgent(ExperienceReplayAgent):
    def __init__(self,
                 env: gym.Env,
                 epochs=10100,
                 epsilon=0.1,
                 N=1000,
                 M=256,
                 num_train_iters=128,
                 num_ep_update_value=2001,
                 decision_interval=8,
                 gamma=1.,
                 layer_size=[64],
                 lr=1e-2):

        super(SlowDownExperienceReplayAgent, self).__init__(env,
                                                            epochs=epochs,
                                                            epsilon=epsilon,
                                                            N=N,
                                                            M=M,
                                                            num_train_iters=num_train_iters,
                                                            num_ep_update_value=num_ep_update_value,
                                                            gamma=gamma,
                                                            layer_size=layer_size,
                                                            lr=lr)
        self.decision_interval = decision_interval
        self.running_reward = 0
        self.curr_action = self.env.action_space.sample()

    def log(self):
        super(SlowDownExperienceReplayAgent, self).log()
        print_and_log ("decision_interval=%d" % self.decision_interval)

    def getAction(self, state):
        '''

        '''
        if (self.t) % self.decision_interval == 0:
            self.curr_action = super(
                SlowDownExperienceReplayAgent, self).getAction(state)
        return self.curr_action

    def update(self, state, action, reward, new_state, timestep, terminal_state):
        '''

        '''
        self.running_reward = reward * \
            self.gamma ** ((timestep + 1) %
                           self.decision_interval) + self.running_reward
        if (timestep) % self.decision_interval == 0:
            self.curr_state = state

        if (timestep + 1) % self.decision_interval == 0:
            self.experience.append(
                (self.curr_state, action, self.running_reward, new_state, terminal_state))
            self.running_reward = 0

    def set_log_filename(self):
        global filename
        filename = "results/%d_%f_%d_%d_%d_%d_%d_%f_%s_%f.log" % (
                 self.epochs,
                 self.epsilon,
                 self.N,
                 self.M,
                 self.num_train_iters,
                 self.num_ep_update_value,
                 self.decision_interval,
                 self.gamma,
                 str(self.layer_size),
                 self.lr)

class FittedQIterationAgent(ExperienceReplayAgent):
    def __init__(self, env: gym.Env, epochs=6000, epsilon=0.1, N=100, M=20, gamma=1.):
        super(FittedQIterationAgent, self).__init__(env, epochs=epochs, epsilon=epsilon,
                                                    N=N, M=M, gamma=gamma)
        # self.Q = DecisionTreeRegressor()
        # self.Q = SGDRegressor()
        self.Q = LinearRegression()
        self.s = StandardScaler()
        self.stateSize = np.prod(self.env.observation_space.shape)

    def getQValue(self, state, action):
        # import pdb; pdb.set_trace()
        try:
            x = self.s.transform([np.append(state, action)])
            # import pdb; pdb.set_trace()
            q = self.Q.predict(x)[0]
        except:
            q = 0
        return q

    def episodeEnded(self):
        '''

        '''
        self.num_episodes += 1
        if (self.num_episodes % self.N == 0):
            print_and_log("Updating Q")
            self.batch_update()

    def batch_update(self):
        # state, action, reward, new_state, terminal_state = self.experience

        model = self.Q

        X = np.array(
            list(map(lambda x: np.append(x[0], x[1]), self.experience)))
        rewards = np.array(list(map(lambda x: x[2], self.experience)))

        model = model.fit(X, rewards)

        X = self.s.fit_transform(X)

        for i in range(self.M):
            allX = np.array(list(map(lambda x: [np.append(x[0], a) for a in range(
                self.env.action_space.n)], self.experience)))
            allXshape = allX.shape

            allX = np.reshape(allX, (-1, allXshape[2]))
            qAllX = model.predict(self.s.transform(allX))
            qMaxX = np.reshape(qAllX, (-1, allXshape[1])).max(axis=1)
            # import pdb; pdb.set_trace()
            y = rewards + self.gamma * qMaxX
            model = model.fit(X, y)

        self.Q = model


class ReinforceAgent(BaseAgent):
    def __init__(self, env: gym.Env, epochs=4000, epsilon=0.05, N=100, M=150, num_tilings=32, tile_dim=8, gamma=1.):
        super(ReinforceAgent, self).__init__(env, epochs=epochs)
        self.experience = []
        self.epsilon = epsilon
        self.num_episodes = 0

        self.N = N
        self.M = M

        self.num_tilings = num_tilings

        self.tile_dim = tile_dim
        self.grid_size = tile_dim ** np.prod(self.env.observation_space.shape)

        self.policy_weights = np.zeros(
            (num_tilings * self.grid_size, self.env.action_space.n))
        self.gradients = np.zeros(
            (num_tilings * self.grid_size, self.env.action_space.n))
        self.value_weights = np.zeros(num_tilings * self.grid_size)
        self.scale = tile_dim / \
            (self.env.observation_space.high - self.env.observation_space.low)

        self.alpdecay = 1
        self.alpha = 0.2
        self.gamma = gamma

    def getAction(self, state):
        '''

        '''
        if np.random.binomial(1, self.epsilon) == 0:
            return np.argmax([np.exp(np.dot(self.getFeaturesPolicy(state, action), self.policy_weights)) for action in range(self.env.action_space.n)])
        else:
            return self.env.action_space.sample()

    def getValue(self, state):
        return np.dot(self.getFeaturesValue(state), self.value_weights)

    def updatePolicy(self, state, action, reward, new_state):
        feat = self.getFeaturesPolicy(state, action)

        gradient = np.zeros(self.num_tilings * self.grid_size)
        denominator = 0
        for actionDash in range(self.env.action_space.n):
            feat_actionDash = self.getFeaturesPolicy(state, actionDash)
            X_difference = feat - feat_actionDash

            temp = np.exp(
                np.dot(feat_actionDash, self.policy_weights[:, actionDash]))
            gradient += temp * X_difference
            denominator += temp

        gradient = gradient / denominator

        self.gradients[:, action] += self.alpdecay * \
            (reward + self.gamma * self.getValue(new_state) -
             self.getValue(state)) * gradient

    def updateValue(self, state, action, reward, new_state):
        self.value_weights += self.alpdecay * (reward + self.gamma * self.getValue(
            new_state) - self.getValue(state)) * self.getFeaturesValue(state)

    def update(self, state, action, reward, new_state, timestep, terminal_state):
        '''

        '''
        if not terminal_state:
            self.updateValue(state, action, reward, new_state)
            if timestep % 10 == 0:
                self.updatePolicy(state, action, reward, new_state)

    def episodeEnded(self):
        '''

        '''
        self.num_episodes += 1
        self.policy_weights += self.gradients
        self.gradients[:, :] = 0

    def getFeaturesValue(self, state):
        feat = np.zeros(self.grid_size * self.num_tilings)
        tileIndex = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale)
        feat[tileIndex] = 1

        return feat

    def getFeaturesPolicy(self, state, action):
        feat = np.zeros(self.grid_size * self.num_tilings)
        tileIndex = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale, [action])
        feat[tileIndex] = 1

        return feat
