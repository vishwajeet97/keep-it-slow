import numpy as np
import gym 
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
                import pdb; pdb.set_trace()
                action = self.getAction(self.state)
                new_state, reward, done, info = self.env.step(action)
                self.update(self.state, action, reward, new_state, t)
                self.state = new_state
                t += 1
                if done:
                    self.episodeEnded()
                    print("Episode finished after {} timesteps".format(t+1))
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
    def __init__(self, env: gym.Env, epsilon = 0.1, N = 10):
        super(ExperienceReplayAgent, self).__init__(env)
        self.experience = []
        self.Q = np.zeros(1)
        self.epsilon = epsilon
        self.num_episodes = 0
        self.N = N
    
    def getAction(self, state):
        '''

        '''
        if np.random.binomial(1, self.epsilon) == 0:
            return np.argmax(self.Q[state, :])
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
            self.batch_update()
    
    def batch_update(self):
        state, action, reward, new_state = np.random.choice(self.experience)
        self.Q[state, ]