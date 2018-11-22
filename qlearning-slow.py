"""
Q-Learning example using OpenAI gym MountainCar enviornment

Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import numpy as np

import gym
from gym import wrappers

n_states = 40
iter_max = 6000

initial_lr = 1.0 # Learning rate
min_lr = 0.03
gamma = 1.0
t_max = 6000
eps = 0.03
d = 5 # Slowing down
d_test = d
iter_test = 1000
def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for i in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            if (i%d_test) == 0:
                action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    # env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print ('----- using Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        current_action = 0
        accum_reward = 0
        prev_state_a, prev_state_b = 0,0
        prev_action = 0
        for j in range(t_max):
            if((j+1)%d==0):
                a, b = obs_to_state(env, obs)
                if np.random.uniform(0, 1) < eps:
                    action = np.random.choice(env.action_space.n)
                    current_action = action
                else:
                    logits = q_table[a][b]
                    logits_exp = np.exp(logits)
                    # import pdb; pdb.set_trace()
                    # if (i%100==0):
                        # print logits_exp
                    probs = logits_exp / np.sum(logits_exp)
                    action = np.random.choice(env.action_space.n, p=probs)
                    current_action = action
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                # update q table
                a_, b_ = obs_to_state(env, obs)
                # q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
                q_table[prev_state_a][prev_state_b][prev_action] = q_table[prev_state_a][prev_state_b][prev_action] + eta * (accum_reward + gamma*np.amax(q_table[a][b]) - q_table[prev_state_a][prev_state_b][prev_action])
                
                accum_reward = reward

                if done:
                    break
            else:
                a, b = obs_to_state(env, obs)
                if (j%d==0):
                    prev_state_a, prev_state_b = a, b
                    prev_action = current_action
                obs, reward, done, _ = env.step(current_action)
                total_reward += reward
                accum_reward += reward
                # update q table
                a_, b_ = obs_to_state(env, obs)
                # q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
                if done:
                    break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(iter_test)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, solution_policy, True)
