import numpy as np
import gym
import tiles
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

class BaseAgent:
    def __init__(self, env, epochs=4000):
        self.env = env
        self.epochs = epochs

    def learn(self):
        ''' 

        '''
        for i in range(self.epochs):
            self.state = self.env.reset()
            self.t = 0
            while True:
                # self.env.render()
                # import pdb
                # pdb.set_trace()
                action = self.getAction(self.state)
                new_state, reward, done, info = self.env.step(action)
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
        raise NotImplementedError

    def update(self, state, action, reward, new_state, timestep, terminal_state):
        '''

        '''
        raise NotImplementedError

    def episodeEnded(self):
        pass


class ExperienceReplayAgent(BaseAgent):
    def __init__(self, env: gym.Env, epochs=4000, epsilon=0.05, N=100, M=150, num_tilings=32, tile_dim=8, gamma=1.):
        super(ExperienceReplayAgent, self).__init__(env, epochs=epochs)
        self.experience = []
        self.epsilon = epsilon
        self.num_episodes = 0

        self.N = N
        self.M = M

        self.num_tilings = num_tilings

        self.tile_dim = tile_dim
        self.grid_size = tile_dim ** np.prod(self.env.observation_space.shape)

        self.weights = np.zeros(
            (self.env.action_space.n, num_tilings * self.grid_size))
        self.scale = tile_dim / \
            (self.env.observation_space.high - self.env.observation_space.low)

        self.alpdecay = 1
        self.alpha = 0.2
        self.gamma = gamma

    def getAction(self, state):
        '''

        '''
        if np.random.binomial(1, self.epsilon) == 0:
            return np.argmax([self.getQValue(state, action) for action in range(self.env.action_space.n)])
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
            for i in range(self.M):
                self.batch_update()
            # self.experience.clear()

    def getFeatures(self, state, action):
        feat = np.zeros(self.grid_size * self.num_tilings)
        tileIndex = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale)
        feat[tileIndex] = 1
        return feat

    def getQValue(self, state, action):
        tileIndices = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale)
        return np.sum(self.weights[action, tileIndices])

    def batch_update(self):

        state, action, reward, new_state, terminal_state = self.experience[np.random.randint(
            len(self.experience))]
        phi_k = self.getFeatures(state, action)
        Q_k = self.getQValue(state, action)

        if terminal_state:
            Q_k_plus_1_max = 0.
        else:
            Q_k_plus_1_max = np.max([self.getQValue(new_state, a)
                                     for a in range(self.env.action_space.n)])

        delta_k = reward + self.gamma * Q_k_plus_1_max - Q_k
        # import pdb; pdb.set_trace()
        # if delta_k != 1.0:
        #     print ("adf")

        # self.weights[action, :] = self.weights[action, :] + (self.alpha) * (delta_k) * phi_k
        self.weights[action, :] = self.weights[action, :] + \
            (1 / (1 + self.alpdecay)) * (delta_k) * phi_k
        self.alpdecay += 0.2


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

        X = np.array(list(map (lambda x: np.append(x[0], x[1]), self.experience)))
        rewards = np.array(list(map (lambda x: x[2], self.experience)))


        model = model.fit(X, rewards)

        X = self.s.fit_transform(X)

        for i in range(self.M):
            allX = np.array(list(map (lambda x: [np.append(x[0], a) for a in range(self.env.action_space.n)], self.experience))) 
            allXshape = allX.shape

            allX = np.reshape(allX, (-1, allXshape[2]))
            qAllX = model.predict(self.s.transform(allX))
            qMaxX = np.reshape(qAllX, (-1, allXshape[1]) ).max(axis=1)
            # import pdb; pdb.set_trace()
            y = rewards + self.gamma * qMaxX
            model = model.fit(X, y)
        
        self.Q = model