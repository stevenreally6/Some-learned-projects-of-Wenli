import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class DeepQNetwork:
    def __init__(
            self,
            actions, features, learning_rate=0.01, gamma=0.95, e=0.95,replace_target_iter=300,
            mem_size=500, batch_size=32, e_greedy_increment=None, output_graph=False,):
        self.actions = actions
        self.features = features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.emax = e
        self.replace_target_iter = replace_target_iter
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.emax

        # total learning step
        self.learn_step = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.mem_size, features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_network()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
        writer.add_graph(graph=tf.get_default_graph())
        
        self.ch = []
        self.c_av=[]

    def learn(self):
        c=0
        # check to replace target parameters
        if self.learn_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nParams_Learned\n')
                
        ##
        # sample batch memory from all memory
        if self.mem_coun > self.mem_size:
            sample_index = np.random.choice(self.mem_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.mem_coun, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.features:],  # fixed params
                self.s: batch_memory[:, :self.features],  # newest params
            })
        ##
        
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.features].astype(int)
        reward = batch_memory[:, self.features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        ##

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.features],
                                                self.q_target: q_target})
        self.ch.append(self.cost)
        ##
        if self.learn_step % self.replace_target_iter == 0 and len(self.ch)>=self.replace_target_iter:
            resi=int(self.learn_step / self.replace_target_iter)
            for i in range((resi-1)*self.replace_target_iter , resi*self.replace_target_iter): 
                c=c+self.ch[i]
            av=c/self.replace_target_iter
            self.c_av.append(av)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.emax else self.emax
        self.learn_step += 1
        ##
        
    def store(self, s, a, r, s_):
        if not hasattr(self, 'mem_coun'):
            self.mem_coun = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.mem_coun % self.mem_size
        self.memory[index, :] = transition

        self.mem_coun += 1
        
    def _build_network(self):
        
        # build evaluate_network
        self.s = tf.placeholder(tf.float32, [None, self.features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

                
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        
        # build target_network
        self.s_ = tf.placeholder(tf.float32, [None, self.features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
        

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.actions)
        return action



    def plot_cost(self): #plot cost in every training
        plt.plot(np.arange(len(self.ch)), self.ch)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        plt.plot(np.arange(len(self.c_av)), self.c_av)
        plt.ylabel('Average Cost in learning interval')
        plt.xlabel('training steps')
        plt.show()
        
        

#%%

#play cart-pole model without control
env = gym.make('CartPole-v0')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

#%%
env = gym.make('CartPole-v0') 
#setting environment, import cart-pole model from gym
env = env.unwrapped

print(env.action_space)  #1x2 action
print(env.observation_space) #1x4 
print(env.observation_space.high)  #1x4 up limit 
print(env.observation_space.low)   #1x4 down limit

RL = DeepQNetwork(actions=env.action_space.n,
                  features=env.observation_space.shape[0],
                  learning_rate=0.01, e=0.9,
                  replace_target_iter=100, mem_size=2000,
                  e_greedy_increment=0.001,) #env.observation_space[0] is current state

total_steps = 0  #total training steps

Episodes=100 #define times of learning episodes
R=[]
All=[]
Ep=[]
for i in range(Episodes):
    All_steps=0 #ALL_steps of every episode (training number)
    observation = env.reset() #initialize state
    ep_r = 0  #total reward of every episode
    while True:
        env.render() #rebuild environment

        action = RL.choose_action(observation) 
        #according to current state and environment, choose better action

        observation_new, reward, done, info = env.step(action) 
        # implement the choosed action to get[state, reward, done, other info]

        # the smaller theta and closer to center is  better and record better reward
        x, x_dot, theta, theta_dot = observation_new
        e1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        e2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = e1 + e2
        RL.store(observation, action, reward, observation_new)
        ep_r += reward
        
        if total_steps > 1000:
            RL.learn()  #when total step is over 1000, start to learn
            
        if done:
            print('episode num: ', i,
                  'eposide reward: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  'total_steps of the episode:',All_steps)
            break 
            #if eposide of current time finish or interrupt, print data of this episode
        
        observation = observation_new
        total_steps += 1
        All_steps += 1 
    
    R.append(ep_r)  #the reward of every eposide
    Ep.append(RL.epsilon)
    All.append(All_steps) #the training number of every eposide
RL.plot_cost() # plot cost of all training

plt.plot(np.arange(len(R)), R) #plot reward of every eposide
plt.ylabel('Reward of every eposide')
plt.xlabel('Eposide')
plt.show()

plt.plot(np.arange(len(All)), All) #plot step number of every eposide
plt.ylabel('Step number of every eposide')
plt.xlabel('Eposide')
plt.show()

plt.plot(np.arange(len(Ep)), Ep) #plot step number of every eposide
plt.ylabel('Epsilon of every eposide')
plt.xlabel('Eposide')
plt.show()

env.close()