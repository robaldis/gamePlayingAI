import gym
import matplotlib.pyplot as plt

env_name = "gym_gs:BreakwallNoFrameskip-v1"
e = gym.make(env_name)
e.reset()
e.render()

# What actions can we do
print(e.action_space)

for i in range(1000):
    s,a,d,i = e.step(1) # 1 is the action

    # r = s[:,:,0]
    # g = s[:,:,1]
    # b = s[:,:,2]

    e.render()

