# run our best trained agent on random games
from stable_baselines import DDPG
from pybulletSim import init_simulation, end_simulation
from env import Drone1DEnv

drone, marker = init_simulation(render = False)
env = Drone1DEnv(drone, marker)
model = DDPG.load('models/ddpg_stablebaseline_1.zip')    # path to the best model


obs = env.reset()
print('initial state: ', obs)

for _ in range(10000):
    action = model.predict(obs)
    obs, rew, done, info = env.step(action[0])

    if done:
        env.reset()

end_simulation()