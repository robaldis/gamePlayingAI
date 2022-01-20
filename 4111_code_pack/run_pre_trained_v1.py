# Script to test a pre-trained model
# Written by Matthew Yee-King
# MIT license 
# https://mit-license.org/

import sys
import os
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import random
import time 

env_name = "gym_gs:BreakwallNoFrameskip-v1" 
model_file = "./pre-trained/mac_hard_breakwall/gym_gs:BreakwallNoFrameskip-v1_20211018-114642_5424"

def create_q_model(num_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)    
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)    
    action = layers.Dense(num_actions, activation="linear")(layer5)    
    return keras.Model(inputs=inputs, outputs=action)

def create_env(env_name, seed=42):
    try:
        # Use the Baseline Atari environment because of Deepmind helper functions
        env = make_atari(env_name)
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        print("Loaded gym")
        env.seed(seed)
        return env
    except:
        print("Failed to make gym env", env_name)
        return None

def run_sim(env, model, frame_count):
    state = np.array(env.reset())
    total_reward = 0
    for i in range(frame_count):
        env.render('human')
        state_tensor = keras.backend.constant(state)
        state_tensor = keras.backend.expand_dims(state_tensor, 0)
        action_values = model(state_tensor, training=False)
        # Take best action
        action = keras.backend.argmax(action_values[0]).numpy()
        state, reward, done, _ = env.step(action)
        state =  np.array(state)
        total_reward += reward
        if done:
            print("Game over at frame", i, "rew", total_reward)
            env.reset()
            #break
        #time.sleep(0.1)
    print("Sim ended : rew is ", total_reward)

def main(env_name, model_file,frame_count=1000,  seed=42):
    env = create_env(env_name=env_name)
    assert env is not None, "Failed to make env " + env_name
    model = create_q_model(num_actions=env.action_space.n)
    model_testfile = model_file + ".data-00000-of-00001"
    assert os.path.exists(model_testfile), "Failed to load model: " + model_testfile
    print("Model weights look loadable", model_testfile)
    model.load_weights(model_file)
    print("Model loaded weights - starting sim")
    run_sim(env, model, frame_count)
        
main(env_name, model_file, frame_count=1000)

