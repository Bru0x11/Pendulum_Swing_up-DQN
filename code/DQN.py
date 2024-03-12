import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from collections import deque
from pendulum import *
import wandb
import os
import matplotlib.pyplot as plt
# import ex_0_policy_evaluation_sol_prof
# from dpendulum import DPendulum

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class DQN:
    def __init__(self, nx, nu):
        self.nx = nx
        self.nu = nu
        self.max_size = 100000
        self.memory_counter = 0
        self.scrolling_index = 0
        self.replay_memory = np.zeros(self.max_size, dtype=object)
        self.batch_size = 64
        self.DISCOUNT = 0.99 #gamma
        self.initial_exploration = 1
        self.final_exploration = 0.001
        self.exploration_decay = 0.0001
        #self.QVALUE_LEARNING_RATE = 1e-3
        #self.optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

        self.update_counter = 0

        self.umin = -4
        self.umax = 4
        self.size_action_space = 21
        self.actions = np.linspace(self.umin, self.umax, self.size_action_space)
        print("Possible actions: ", self.actions)

        self.Q = self.get_critic(self.nx, self.nu)
        self.target_Q = self.get_critic(self.nx, self.nu)


    def get_critic(self, nx, nu):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(nx))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.size_action_space)(state_out4) 

        model = tf.keras.Model(inputs, outputs)

        return model

    def save_experience(self, x, u, c, next_x, is_last):
        index = self.scrolling_index % self.max_size
        self.replay_memory[index] = (x, u, c, next_x, is_last)
        self.scrolling_index += 1
        if self.memory_counter != self.max_size:
            self.memory_counter += 1

    def get_epsilon_action(self, x, episode, exp = True):

        if exp:
            #exponential
            epsilon = self.final_exploration + (self.initial_exploration-self.final_exploration) * np.exp(-self.exploration_decay*episode)
        else:
            #linear
            epsilon = max(self.final_exploration,self.initial_exploration - self.exploration_decay * episode)

        if(np.random.rand(1)<=epsilon):
            idx_u = np.random.choice(self.size_action_space)
        else:
            input = np.expand_dims(np.array(x), axis=0)
            q_values = self.Q(input)
            idx_u = np.argmax(q_values)

        return idx_u
    
    def fill_replay_memory(self, env):
        for i in range(1000):
            index = self.scrolling_index % self.max_size
            x = env.reset()
            #todo: better 
            if self.nx == 2:   
                x = [x[0][0], x[1][0]]
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            idx_u = np.random.choice(self.size_action_space)
            u = self.actions[idx_u]
            next_x, cost = env.step(u)
            
            if self.nx == 2:   
                next_x = [next_x[0][0], next_x[1][0]]

            else: 
                next_x = [next_x[0][0], next_x[0][1], next_x[1][0], next_x[1][1]]

            is_last = False
            t = (x, idx_u, cost, next_x, is_last)
            self.replay_memory[index] = t
            self.scrolling_index += 1
            if self.memory_counter != self.max_size:
                self.memory_counter += 1


def train_dqn(dqn, env, episodes, critic_optimizer, save_dir, go_on=False):
    
    wandb.login(key='3ec27bc14641f193a2a62ed18f343b1fba35484a')
    wandb.init(project="pendulum_DQN")

    if go_on:
        Q = tf.keras.models.load_model(save_dir)
        Q_target = dqn.target_Q
    else:
        Q = dqn.Q
        Q_target = dqn.target_Q

    Q.summary()

    # Set initial weights of targets equal to those of the critic
    Q_target.set_weights(Q.get_weights())

    dqn.fill_replay_memory(env)
 
    total_reward_hist = []
    avg_reward_hist = []
    for episode in range(1, episodes+1):
        #env.reset(np.array([np.pi,0.,0.,0.]))
        if env.nx == 4: 
            env.reset(np.array([[np.pi, 0.],[0.,0.]]))
            #env.reset()
        else:
            #env.reset()
            env.reset(np.array([[np.pi],[0.]]))
        n_steps = 400
        sum_costs = 0 
        for t in range(1,n_steps+1):
            x = env.x
            if env.nx == 2:
                x = [x[0][0], x[1][0]]
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            idx_u = dqn.get_epsilon_action(x, episode)
            u = dqn.actions[idx_u]

            next_x, cost = env.step(u)
            if env.nx == 2:
                next_x = [next_x[0][0], next_x[1][0]]

            else:
                next_x = [next_x[0][0], next_x[0][1], next_x[1][0], next_x[1][1]]
            is_last = False
            if t==n_steps:
                is_last = True

            dqn.save_experience(x, idx_u, cost, next_x, is_last)
            
            if((dqn.update_counter % 4) == 0):
                indexes_batch = np.random.choice(dqn.memory_counter-1, dqn.batch_size, replace=False).tolist()
                current_batch = dqn.replay_memory[indexes_batch]
                states = []
                actions = []
                costs = []
                next_states = []
                is_lasts = []

                for t in current_batch:
                    states.append(t[0])
                    actions.append(t[1])
                    costs.append(t[2])
                    next_states.append(t[3])
                    is_lasts.append(t[4])

                indices = np.arange(dqn.batch_size)

                with tf.GradientTape() as tape:         
                    q_pred = Q(np.array(states), training = True)[indices, actions]
                    q_target_pred = Q_target(np.array(next_states))

                    q_target_pred = np.array(np.max(q_target_pred, axis=1))

                    q_target_pred[is_lasts] = 0.0

                    total_cost = costs + (dqn.DISCOUNT * q_target_pred)   
                    # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
                    Q_loss = tf.math.reduce_mean(tf.math.square(total_cost - q_pred))  

                # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
                Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
                # Update the critic backpropagating the gradients
                critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))

            sum_costs += cost
            
            #todo
            if(is_last):
                break
            
            if((dqn.update_counter % 2500) == 0):
                Q_target.set_weights(Q.get_weights())

            dqn.update_counter += 1

        total_reward_hist.append(sum_costs)
        avg_reward = np.average(total_reward_hist[:])
        avg_reward_hist.append(avg_reward)
        epsilon = dqn.final_exploration + (dqn.initial_exploration-dqn.final_exploration) * np.exp(-dqn.exploration_decay*episode)
        wandb.log({'episode': episode, 'total_reward': sum_costs, 'avg_reward': avg_reward, 'epsilon' : epsilon })

        if episode%250 == 0:
            temp_save_dir = save_dir + f"{episode}_episode"
            print("Saving model...")
            Q.save(temp_save_dir)
            print(temp_save_dir)

        print("Episode :", episode, "sum_costs : {:.2f}".format(sum_costs), "Avg Reward : {:.2f}".format(avg_reward))
 

    Q.save(save_dir)
    wandb.finish()
    fig, ax = plt.subplots()
    t = np.arange(episodes)
    ax.plot(t, total_reward_hist, label="Total Reward")
    ax.plot(t, avg_reward_hist, label="Average Reward")
    ax.set_title("Reward vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()



def test_dqn(env, episodes, save_dir, plot=True):
    Q = tf.keras.models.load_model(save_dir)
    total_reward_hist = []
    avg_reward_hist = []

    for episode in range(1, episodes+1):
        print("Episode: ", episode)
        if env.nx == 4: 
            #env.reset(np.array([[-np.pi, 0.],[0.,0.]]))
            env.reset()
        else:
            #env.reset(np.array([[np.pi],[0.]]))
            env.reset()
        n_steps = 400
        tot_reward = 0 
        
        angles = []
        velocities = []

        angles_first = []
        angles_second = []
        velocities_first = []
        velocities_second = []

        torque = []
        for t in range(1,n_steps+1):
            x = env.x
            
            if env.nx == 2:
                x = [x[0][0], x[1][0]]
                angles.append(x[0])
                velocities.append(x[1])
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]
                angles_first.append(x[0])
                angles_second.append(x[1])
                velocities_first.append(x[2])
                velocities_second.append(x[3])

            input = np.expand_dims(np.array(x), axis=0)
            q_values = Q(input)
            idx_u = np.argmax(q_values)
            u = dqn.actions[idx_u]
            torque.append(u)
            next_x, reward = env.step(u)
            tot_reward += reward
            if(t==n_steps):
                last_reward = reward


        if plot:
            

            plt.rcParams["figure.figsize"] = [10, 7]
            plt.rcParams["figure.autolayout"] = True

            if env.nx == 2:
                timesteps = np.arange(len(angles))
                fig, (ax1, ax2) = plt.subplots(2, 1)

                ax1.set_title("State graph")
                ax1.plot(timesteps, angles, color="red", label="angles")
                ax1.plot(timesteps, velocities, color="blue", label="velocities")
                ax1.legend()

                ax2.set_title("Torque graph")
                ax2.plot(timesteps, torque, color="blue", label="torque")
                ax2.legend()

                plt.show()
            else:
                timesteps = np.arange(len(angles_first))
                fig, (ax1, ax2) = plt.subplots(2, 1)

                ax1.set_title("State graph")
                ax1.plot(timesteps, angles_first, color="#1f77b4", label="angles first joint")
                ax1.plot(timesteps, angles_second, color="#ff7f0e", label="angles second joint")

                ax1.plot(timesteps, velocities_first, color="#2ca02c", label="velocities first joint")
                ax1.plot(timesteps, velocities_second, color="#d62728", label="velocities second joint")
                ax1.legend()

                ax2.set_title("Torque graph")
                ax2.plot(timesteps, torque, color="blue", label="torque")
                ax2.legend()

                plt.show()


        total_reward_hist.append(tot_reward)
        avg_reward = np.average(total_reward_hist[:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "total_reward : {:.2f}".format(tot_reward), "Avg Reward : {:.2f}".format(avg_reward))


def pi(env, x):
    save_dir = "models/single_pendulum/prof_reward/1000_episodes1000_episode_random"
    Q = tf.keras.models.load_model(save_dir)

    x = env.x
            
    # if env.nx == 2:
    #     x = [x[0][0], x[1][0]]
    # else:
    #     x = [x[0][0], x[0][1], x[1][0], x[1][1]]

    print(x)

    input = np.expand_dims(np.array(x), axis=0)
    print(input)
    print(np.shape(input))
    q_values = Q(input)
    idx_u = np.argmax(q_values)
    u = dqn.actions[idx_u]

    return u


if __name__ == "__main__":

    # Set the WANDB__EXECUTABLE environment variable to the path to a valid Python interpreter.
    os.environ["WANDB__EXECUTABLE"] = "/usr/bin/python3"

    njoints = 2
    episodes = 100
    QVALUE_LEARNING_RATE = 1e-3

    save_dir = "models/"
    if njoints == 1:
        save_dir += "single_pendulum/"
    else:
        save_dir += "double_pendulum/"
    save_dir += f"{episodes}_episodes"

    env = Pendulum(njoints)

    # Create critic and target NNs
    if njoints == 1:
        nx = 2
        nu = 1
    else:
        nx = 4
        nu = 2
    
    dqn = DQN(nx, nu)
    critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

    #train_dqn(dqn, env, episodes, critic_optimizer, save_dir, go_on=False)

    test_dir = "models/double_pendulum/10000_episodes10000_episode"
    test_dqn(env, episodes, test_dir)



    # ### --- Hyper paramaters
    # MAX_ITERS         = 200         # Max number of iterations
    # CONVERGENCE_THR   = 1e-4        # convergence threshold
    # NPRINT            = 5           # Print info every NPRINT iterations
    # PLOT              = True        # Plot the V table
    # DISCOUNT          = 0.99         # Discount factor 

    # ### --- Environment
    # nq=21   # number of discretization steps for the joint angle q
    # nv=11   # number of discretization steps for the joint velocity v
    # nu=11   # number of discretization steps for the joint torque u
    # env = Pendulum(1) # create the environment
    # V  = np.zeros([env.nx])     # V-table initialized to 0

    # V = np.zeros([nq*nv, 1])
    # print(V)
    # print(np.shape(V))
    # V = ex_0_policy_evaluation_sol_prof.policy_eval(env, DISCOUNT, pi, V, MAX_ITERS, CONVERGENCE_THR, nq, nv, PLOT, NPRINT)
        
