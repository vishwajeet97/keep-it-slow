"""
Q-Learning example using OpenAI gym MountainCar enviornment

Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import numpy as np
from constants import *
import gym
from gym import wrappers
import sys

initial_lr = 1.1 # Learning rate
min_lr = 0.003

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
    d = int(sys.argv[1])
    d_test = d
    # env_name = 'CartPole-v0'
    filename = str(env_name)+"qslowing-"+str(d)+".txt"
    file = open(filename, 'w') 
    env = gym.make(env_name)
    print ('----- using Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    plot_variable = np.zeros((number_of_average_runs,int(t_max/testIter)+1,2))
    for av in range(number_of_average_runs):
        env.seed(av)
        np.random.seed(av)
        q_table = np.zeros((n_states, n_states, 3))
        for i in range(iter_max):
            obs = env.reset()
            total_reward = 0
            ## eta: learning rate is decreased at each step
            eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
            current_action = np.random.choice(env.action_space.n)
            accum_reward = 0
            prev_state_a, prev_state_b = 0,0
            prev_action = 0
            set_flag = 0
            for j in range(t_max):
                if(j%d==0):
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
                    if(set_flag):
                        q_table[prev_state_a][prev_state_b][prev_action] = q_table[prev_state_a][prev_state_b][prev_action] + eta * (accum_reward + gamma*np.amax(q_table[a][b]) - q_table[prev_state_a][prev_state_b][prev_action])
                    
                    prev_state_a = a
                    prev_state_b = b
                    prev_action = current_action
                    if(set_flag==0):
                        set_flag = 1
                    accum_reward = reward

                    if done:
                        break
                else:
                    # a, b = obs_to_state(env, obs)
                    # if (j%d==0):
                    #     # prev_state_a, prev_state_b = a, b
                    #     prev_action = current_action
                    obs, reward, done, _ = env.step(current_action)
                    total_reward += reward
                    accum_reward += reward
                    # update q table
                    # a_, b_ = obs_to_state(env, obs)
                    # q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
                    if done:
                        break
            if i % 100 == 0:
                print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
            if i % testIter == 0:
                solution_policy = np.argmax(q_table, axis=2)
                solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(iter_test)]
                print('Iteration #%d --' %(i+1))
                print("Average score of solution = ", np.mean(solution_policy_scores))
                plot_variable[av][i/testIter][0] = i
                plot_variable[av][i/testIter][1] = np.mean(solution_policy_scores)
                # file.write(str(i))
                # file.write("\t")
                # file.write(str(np.mean(solution_policy_scores)))
                # file.write("\n")
        solution_policy = np.argmax(q_table, axis=2)
        solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(iter_test)]
        ans = np.mean(solution_policy_scores)
        plot_variable[av][-1][0] = iter_max
        plot_variable[av][-1][1] = ans
        print("Average score of solution = ", ans)
        # Animate it

        run_episode(env, solution_policy, True)
    plot_variable = np.mean(plot_variable, axis=0)
    for i in range(int(t_max/testIter)+1):
        file.write(str(plot_variable[i][0]))
        file.write("\t")
        file.write(str(plot_variable[i][1]))
        file.write("\n")
    
    file.close()
