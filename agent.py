import numpy as np
import gym
import tiles
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

from model import Q_Network
import torch
import torch.nn.functional as F


class BaseAgent:
    def __init__(self, env, epochs=4000):
        self.env = env
        self.epochs = epochs

    def learn(self):
        ''' 

        '''
        for i in range(self.epochs):
            self.state = self.env.reset()
            self.state = self.state - self.state
            self.t = 0
            while True:
                if i > 3000: 
                    self.env.render()
                # import pdb
                # pdb.set_trace()
                action = self.getAction(self.state)
                new_state, reward, done, _ = self.env.step(action)
                new_state -= self.state
                self.update(self.state, action, reward,
                            new_state, self.t, done)
                self.state = new_state
                self.t += 1
                if done:
                    self.episodeEnded()
                    print("Episode finished after {} timesteps".format(self.t+1))
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


class ExperienceReplayAgent(BaseAgent):
    def __init__(self,
        env: gym.Env,
        epochs=4000,
        epsilon=0.1,
        N=500,
        M=64,
        num_train_iters=50,
        num_tilings=32,
        tile_dim=8,
        gamma=1.):

        super(ExperienceReplayAgent, self).__init__(env, epochs=epochs)
        self.experience = []
        self.epsilon = epsilon
        self.num_episodes = 0

        self.NUM_EP_UPDATE_VALUE = 10

        self.N = N
        self.M = M
        self.num_train_iters = num_train_iters

        self.Q_new = Q_Network(
            input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.n, layer_size=[2, 2])
        self.Q_old = Q_Network(
            input_dim=self.env.observation_space.shape[0], output_dim=self.env.action_space.n, layer_size=[2, 2])

        self.Q_old.load_state_dict(self.Q_new.state_dict())
        self.Q_old.eval()
        self.optimizer = torch.optim.Adam(self.Q_new.parameters())

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
            print("Updating Q")
            self.batch_update()

        if (self.num_episodes % self.NUM_EP_UPDATE_VALUE == 0):
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
        v_values[torch.Tensor(final.tolist())==False] = self.Q_old(next_state).max(1)[0].detach()

        expected_q_values = v_values * self.gamma + rewards

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        for p in self.Q_new.parameters():
            p.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        # import pdb; pdb.set_trace()
        print([np.sum(np.abs(p.grad.data.numpy()))
               for p in self.Q_new.parameters()])
        # self.Q_new.eval()

    def batch_update(self):
        for i in range(self.num_train_iters):
            batch_index = np.random.randint(len(self.experience), size=self.M)
            exp = np.array(self.experience)[batch_index]

            batch = (np.array(exp[:, 0].tolist()), np.array(exp[:, 1].tolist()), np.array(
                exp[:, 2].tolist()), np.array(exp[:, 3].tolist()), np.array(exp[:, 4].tolist()))
            self.train_replay(batch)


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
            print("Updating Q")
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
