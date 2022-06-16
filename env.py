import numpy as np
import random
import gym
from gym import spaces
import pybullet as p


class Drone1DEnv(gym.Env):
    metadata = {'render.modes':['human']}


    def __init__(self, drone, marker):
        super(Drone1DEnv, self).__init__()

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(2,), 
                                            dtype=np.float32)

        self.action_space = spaces.Box(low = -1, high = 1,
                                       shape = (1,), 
                                       dtype=np.float32)

        self.drone = drone
        self.marker = marker
        self.z_des = 8.0      # m
        self.z_dot_des = 0.0  # m/s
        self.mass = 1.5       # kg
        self.gravity = 9.81   # m/s^2
        self.obs_max = 5.0


    # current state--> current position, current velocity in z-direction
    def state(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        drone_vel, _ = p.getBaseVelocity(self.drone)

        # discretization and error representation of state
        state = self.abs_to_error_state(drone_pos[2], drone_vel[2])
        return state


    # reward
    def reward(self, action):
        e_z, e_z_dot = self.state()
        reward = 0.0
        if self.done():
            if (abs(e_z)>=1.5 or abs(e_z_dot)>=1):
                reward = -5.0
                print('...outbound conditions...', e_z, e_z_dot)
            else:
                reward = 100.0
                print('----desired point achieved----')
        else:            
            reward = -(10*abs(e_z) + 0.5*abs(e_z_dot) + 0.3*abs(action))

        return float(reward)
    
    
    # whether goal is achieved or not.
    def done(self):
        e_z, e_z_dot = self.state()
        '''
        done=1; episodes terminates when:
          1. if e_z >= 1.5: drone is 1.5*5 m or more apart from desired pos.
          2. if e_z_dot >= 1: drone velocity is >=5 m/s {pysical constraints}.
          3. if e_z<=0.01/5.0 and e_z_dot<=0.01/5: desired condition is achieved.
        '''
        if (abs(e_z)>=1.5 or abs(e_z_dot)>=1) or\
            (abs(e_z)<=0.01/self.obs_max\
             and abs(e_z_dot)<=0.01/self.obs_max):
            return True
        return False


    #info
    def info(self):
        return {}


    # step
    def step(self, action):
        # action must be a float.
        action_ = (action+1)*self.mass*self.gravity
        p.applyExternalForce(objectUniqueId=self.drone, linkIndex=-1,
                         forceObj=[0, 0 ,action_], posObj=[0,0,0], 
                         flags=p.LINK_FRAME)
        p.stepSimulation()
        state = self.state()
        reward = self.reward(action)
        done = self.done()
        info = self.info()
        return state, reward, done, info


    # reset the environment
    def reset(self):
        # initializing quadcopter with random z_position and z_velocity
        droneStartPos, droneStartOrn, droneStartLinVel, droneStartAngVel\
             = self.random_state_generator()
        p.resetBasePositionAndOrientation(self.drone, droneStartPos,
                                          droneStartOrn)
        p.resetBaseVelocity(self.drone, droneStartLinVel, 
                            droneStartAngVel)
        # print("\n[--------z_des: %f,    z_init: %f,     v_init: %f ---------]\n\n"
        #       %(self.z_des, droneStartPos[2], droneStartLinVel[2]))

        # return state
        state  = self.abs_to_error_state(droneStartPos[2], droneStartLinVel[2])
        return state


    def random_state_generator(self):
        # initialize drone's position between 3 and 13 m.
        z_init = random.uniform(3.0,13.0) 
        # initialized with velocity in between -1 and 1 m/s.
        z_dot_init = random.uniform(-1,1)
        
        StartPos = [0,0,z_init] 
        StartOrn = p.getQuaternionFromEuler([0,0,0])
        StartLinVel = [0,0,z_dot_init]
        StartAngVel = [0,0,0]
        return StartPos, StartOrn, StartLinVel, StartAngVel


    def abs_to_error_state(self, z, z_dot):
        e_z = (z - self.z_des) / self.obs_max
        e_z_dot = (z_dot - self.z_dot_des) / self.obs_max

        return np.array([e_z, e_z_dot])


    def render(self, mode='human'):
        pass