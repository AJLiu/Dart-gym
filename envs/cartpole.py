import numpy as np
from gym import utils
from envs import dart_env

class DartCartpoleEnv(dart_env.DartEnv):
    def __init__(self):
        control_bounds = np.array([[1.0], [1.0]])
        self.action_scale = 100
        super().__init__('cartpole.skel', 1)

    def step(self, a):
        ctrl = np.zeros(self.ndofs)
        ctrl[0] = np.array(a[0] * self.action_scale)
        self.do_simulation(ctrl, self.frame_skip)

        obs = self.state_vector()
        print(obs)
        reward = 1.0
        done = not np.isfinite(obs).all() or (np.abs(obs[1]) > .2)
        return obs, reward, done, {}

    def reset_model(self):
        self.model.setPositions(self.init_qpos)
        self.model.setVelocities(self.init_qvel)
        return self.state_vector()