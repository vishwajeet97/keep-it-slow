import numpy as np
import gym
import tiles


class BaseAgent:
    def __init__(self, env, epochs=1000):
        self.env = env
        self.epochs = epochs

    def learn(self):
        ''' 

        '''
        for i in range(self.epochs):
            self.state = self.env.reset()
            self.t = 0
            while True:
                self.env.render()
                # import pdb
                # pdb.set_trace()
                action = self.getAction(self.state)
                new_state, reward, done, info = self.env.step(action)
                self.update(self.state, action, reward, new_state, self.t)
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

    def update(self, state, action, reward, new_state, timestep):
        '''

        '''
        raise NotImplementedError

    def episodeEnded(self):
        pass


class ExperienceReplayAgent(BaseAgent):
    def __init__(self, env: gym.Env, epsilon=0.1, N=10, M=100, num_tilings=16, tile_dim=4, gamma = 1.):
        super(ExperienceReplayAgent, self).__init__(env)
        self.experience = []
        self.epsilon = epsilon
        self.num_episodes = 0

        self.N = N
        self.M = M

        self.num_tilings = num_tilings
        
        self.tile_dim = tile_dim
        self.grid_size = tile_dim ** np.prod(self.env.observation_space.shape)

        self.weights = np.random.randn(num_tilings * self.grid_size)
        self.scale = tile_dim / (self.env.observation_space.high - self.env.observation_space.low)

        self.alpha = 0.5
        self.gamma = gamma

    def getAction(self, state):
        '''

        '''
        if np.random.binomial(1, self.epsilon) == 0:
            return np.argmax([self.getQValue(state, action) for action in range(self.env.action_space.n)])
        else:
            return self.env.action_space.sample()

    def update(self, state, action, reward, new_state, timestep):
        '''

        '''
        self.experience.append((state, action, reward, new_state))

    def episodeEnded(self):
        '''

        '''
        self.num_episodes += 1
        if (self.num_episodes % self.N == 0):
            for i in range(self.M):
                self.batch_update()

    def getFeatures(self, state, action):
        feat = np.zeros(self.grid_size * self.num_tilings)
        tileIndex = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale, [action])
        feat[tileIndex] = 1
        return feat

    def getQValue(self, state, action):
        tileIndices = tiles.tiles(
            self.num_tilings, self.grid_size * self.num_tilings, state * self.scale, [action])
        return np.sum(self.weights[tileIndices])

    def batch_update(self):
        state, action, reward, new_state = np.random.choice(self.experience)
        phi_k = self.getFeatures(state, action)
        Q_k = self.getQValue(state, action)

        Q_k_plus_1_max = np.max( [ self.getQValue(new_state, a) for a in range(self.env.action_space.n)] )
        
        delta_k = reward + self.gamma * Q_k_plus_1_max - Q_k

        self.weights = self.weights + self.alpha * (delta_k) * phi_k
