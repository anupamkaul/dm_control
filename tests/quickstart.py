from dm_control import suite
import numpy as np
import time

# this is an RL equivalent example of how to set up RL with dm_control
# which also talks to mujoco's models (humanoid etc)

# load one task:
env = suite.load(domain_name = "cartpole", task_name = "swingup")

# iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
    env = suite.load(domain_name, task_name)
    print("domain name = ", domain_name, " task_name = ", task_name)
    time.sleep(2)

    # step through an episode and print out reward, discount and observation
    action_spec = env.action_spec()
    time_step = env.reset()

    while not time_step.last():
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        time_step = env.step(action)
        print(time_step.reward, time_step.discount, time_step.observation)
        #time.sleep(1)

