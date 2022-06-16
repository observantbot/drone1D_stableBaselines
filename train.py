# train a DDPG algorithm based model using Stable Baselines library from open AI.

import numpy as np
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from pybulletSim import init_simulation, end_simulation
from env import Drone1DEnv


drone, marker = init_simulation()
env = Drone1DEnv(drone, marker)
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=float(0.5) * np.ones(1))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise,
             action_noise=action_noise)

for i in range(1, 51):
  save_path = "models/ddpg_stablebaseline_" + str(i) + ".zip"
  if i==1:
    model.learn(total_timesteps=10000)
    # at every 10000 time steps, we will save our model
    model.save(save_path)
  else:
    del model
    model = DDPG.load(prev_path, env)
    model.learn(total_timesteps=10000)
    model.save(save_path)
  prev_path = save_path

print('done')

end_simulation()