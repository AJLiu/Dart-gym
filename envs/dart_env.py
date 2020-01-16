from os import path
from multiprocessing import Process, Value, Manager
import time

from gym import error, spaces
from gym.utils import seeding
import numpy as np
import gym
import dartpy as dart

DEFAULT_SIZE = 500

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def viewer_thread(viewer):
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([2.0, 1.0, 2.0],
                                [0.0, 0.0, 0.0],
                                [-0.24, 0.94, -0.25])
    viewer.run()


class EnvNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, skel):
        super(EnvNode, self).__init__(world)
        self.forces = np.zeros(skel.getNumDofs())

    def customPreStep(self):
        self.skel.setForces(self.forces)

    def set_forces(self, ctrl):
        self.forces = ctrl


class DartEnv(gym.Env):
    def __init__(self, model_path, frame_skip):
        super().__init__()

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.frame_skip = frame_skip
        self.viewer = None
        
        self.world = dart.utils.SkelParser.readWorld(fullpath)
        num_skeletons = self.world.getNumSkeletons()
        self.model = self.world.getSkeleton(num_skeletons-1)
        
        self.init_qpos = self.model.getPositions()
        self.init_qvel = self.model.getVelocities()
        self.ndofs = self.model.getNumDofs()

        # Set Action Space
        low, high = self.model.getForceLowerLimits(), self.model.getForceUpperLimits()
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Set Observation Space
        action = self.action_space.sample()
        observation, _, done, _ = self.step(action)
        assert not done
        self.observation_space = convert_observation_to_space(observation)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # -------------------------------------------------------------------------
    # Override these functions
    # -------------------------------------------------------------------------
    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass
    # -------------------------------------------------------------------------

    def reset(self):
        self.world.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.ndofs,) and qvel.shape == (self.ndofs,)
        self.model.setPositions(qpos)
        self.model.setVelocities(qvel)

    @property
    def dt(self):
        return self.world.getTimeStep() * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for _ in range(n_frames):
            self.model.setForces(ctrl)
            self.world.step()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_id=None, camera_name=None):
        if self.viewer is None:
            self.node = EnvNode(self.world, self.model)
            self.viewer = dart.gui.osg.Viewer()
            self.viewer.addWorldNode(self.node)
            # p = Process(target=viewer_thread, args=(self.viewer,))
            # p.start()
            viewer_thread(self.viewer)
        time.sleep(0.10)

    def close(self):
        if self.viewer is not None:
            # self.viewer.close()
            self.viewer = None

    def state_vector(self):
        return np.concatenate([
            self.model.getPositions(),
            self.model.getVelocities()
        ])