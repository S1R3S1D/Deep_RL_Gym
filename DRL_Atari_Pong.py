%pip install -U gym>=0.21.0
%pip install -U gym[atari,accept-rom-license]

import cv2
import numpy as np
from collections import deque
import gym
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def resize_frame(frame):
  frame = frame[30:-12, 5:-4]
  frame = np.average(frame, axis = 2)
  frame = cv2.resize(frame, (84, 84), interpolation = cv2.INTER_NEAREST)
  frame = np.array(frame, dtype = np.uint8)
  return frame


class Memory():
  def __init__(self, max_len):
    self.max_len = max_len
    self.frames = deque(maxlen = max_len)
    self.actions = deque(maxlen = max_len)
    self.rewards = deque(maxlen = max_len)
    self.done_flags = deque(maxlen = max_len)

  def add_experience(self, next_frame, next_frames_reward, next_action, next_frame_terminal):
    self.frames.append(next_frame)
    self.actions.append(next_action)
    self.rewards.append(next_frames_reward)
    self.done_flags.append(next_frame_terminal)


def initialize_new_game(name, env, agent):

  env.reset()
  starting_frame = resize_frame(env.step(0)[0])

  dummy_action = 0
  dummy_reward = 0
  dummy_done = False
  for _ in range(3):
    agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent):
  env = gym.make(name)
  return env

def take_step(name, env, agent, score, debug):

  agent.total_timesteps +=1
  if agent.total_timesteps %50000==0:
    agent.model.save_weights('recent_weights.hdf5')
    print('Weights Saved')

  next_frame, next_frames_reward, next_frame_terminal, info = env.step(agent.memory.actions[-1])

  next_frame = resize_frame(next_frame)
  new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
  new_state = np.moveaxis(new_state, 0, 2)/255
  new_state = np.expand_dims(new_state, 0)

  next_action = agent.get_action(new_state)

  if next_frame_terminal:
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
    return (score + next_frames_reward),True

  agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

  if debug:
    env.render()

  if len(agent.memory.frames) > agent.starting_mem_len:
    agent.learn(debug)

  return (score + next_frames_reward),False

def play_episode(name, env, agent, debug = False):
  initialize_new_game(name, env, agent)
  done = False
  score = 0
  while True:
    score,done = take_step(name,env,agent,score, debug)
    if done:
      break
  return score



class Agent():
  def __init__(self, possible_actions, starting_mem_len, max_mem_len, starting_epsilon, learning_rate, debug = False):
    self.memory = Memory(max_mem_len)
    self.possible_actions = possible_actions
    self.epsilon = starting_epsilon
    self.epsilon_decay = 0.9/100000
    self.epsilon_min = 0.05
    self.gamma = 0.95
    self.learning_rate = learning_rate
    self.model = self.build_model()
    self.model_target = clone_model(self.model)
    self.total_timesteps = 0
    self.starting_mem_len = starting_mem_len
    self.learns = 0

  def build_model(self):
    model = Sequential()
    model.add(Input((84, 84, 84)))
    model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters = 64,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2)))
    model.add(Dense(len(self.possible_actions), activation = 'linear'))
    optimizer = Adam(self.learning_rate)
    model.compile(optimizer, loss=tf.keras.losses.Huber())
    return model

  def get_action(self, state):
    if np.random.rand()<self.epsilon:
      return random.sample(self.possible_actions, 1)[0]

    a_index = np.argmax(self.model.predict(state))
    return self.possible_actions[a_index]

  def _index_valid(self, index):
    if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
      return False
    else:
      return True

  def learn(self, debug = False):
    states = []
    next_states = []
    actions_taken = []
    next_rewards = []
    next_done_flags = []

    while len(states)<32:
      index = np.random.randint(4, len(self.memory.frames)-1)
      if self._index_valid(index):
        state = [self.memory.frames[index-3], self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index]]
        state = np.moveaxis(state,0,2)/255
        next_state = [self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index], self.memory.frames[index+1]]
        next_state = np.moveaxis(next_state,0,2)/255

        states.append(state)
        next_states.append(next_state)
        actions_taken.append(self.memory.actions[index])
        next_rewards.append(self.memory.rewards[index+1])
        next_done_flags.append(self.memory.done_flags[index+1])

    labels = self.model.predict(np.array(states))
    next_state_values = self.model_target.predict(np.array(next_states))

    for i in range(32):
      action = self.possible_actions.index(actions_taken[i])
      labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])

    self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_decay
      self.learns += 1

    if self.learns % 10000 == 0:
      self.model_target.set_weights(self.model.get_weights())
      print('\nTarget model updated')


#%%


name = 'PongDeterministic-v4'

agent = Agent(possible_actions = [0, 2, 3], starting_mem_len = 50000, max_mem_len=750000,starting_epsilon = 1, learning_rate = 0.00025)
env = make_env(name, agent)

last_100_avg  = [-21]
scores = deque(maxlen = 100)
max_score = -21

env.reset()

for i in range(1000000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = play_episode(name, env, agent, debug = False) #set debug to true for rendering
    scores.append(score)
    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if i%100==0 and i!=0:
        last_100_avg.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,100),last_100_avg)
        plt.show()
