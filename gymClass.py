import gym
import numpy as np
# import cv2
from gym import spaces
from collections import deque
import time
# class adjustFrame(gym.ObservationWrapper):
#
#     def __init__(self, env):
#         """
#         Warp frames to 84x84 as done in most atari papers.
#
#         """
#
#         gym.ObservationWrapper.__init__(self, env)
#         self.width = 84
#         self.height = 84
#         self.observation_space = spaces.Box(low=0, high=255,
#             shape=(self.height, self.width, 1), dtype=np.uint8)
#
#     def observation(self, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
#         return frame[:, :,None]
#
#     def close(self):
#         return self.env.close()
#
# class stackFrames(gym.Wrapper):
#     def __init__(self, env, k):
#         """
#         Stack k last frames.
#         This is to let the NN able to extract information about velocity and acceleration.
#         Also normalizes observation ([0-1] instead of [0 255]) for faster converges
#         """
#
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.frames = deque([], maxlen=self.k)
#         shp = env.observation_space.shape
#         self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0], shp[1], shp[2] * self.k), dtype=np.float32)
#
#     def reset(self):
#         ob = self.env.reset().astype(np.float32)/np.float32(255.0)
#         for _ in range(self.k):
#             self.frames.append(ob)
#         return np.concatenate(self.frames,axis=2)
#
#     def step(self, action):
#         ob, reward, done, info = self.env.step(action)
#         self.frames.append(ob.astype(np.float32)/np.float32(255.0))
#         return np.concatenate(self.frames,axis=2), reward, done, info
#
#     def close(self):
#         return self.env.close()


class multiEnv:

    def __init__(self, envL):
        self.envL = envL
        self.numEnvs = len(self.envL)
        self.AllEpR = []
        self.TimeSteps = []
        self.EpisodeLength = []
        self.ElapsedTime = []
        self.StartT = time.time()
        self.observation_space = self.envL[0].observation_space
        self.action_space = self.envL[0].action_space

    def reset(self):
        for env in self.envL:
            env.runningRewards = 0
            env.runningLength = 0
        return np.array([env.reset() for env in self.envL]).reshape(-1,self.observation_space.shape[0])

    def step(self, action, timestep):
        obsL, rewardL, doneL, infoL = [],[],[],[]
        if self.numEnvs==1:
            action = [action]
        for i, (a, env)  in enumerate(zip(action, self.envL)):
            obs, reward, done, info = env.step(a)
            env.runningRewards += reward
            env.runningLength += 1
            if done:
                obs = env.reset()
                self.AllEpR.append(env.runningRewards)
                self.TimeSteps.append(timestep*self.numEnvs)
                self.EpisodeLength.append(env.runningLength)
                self.ElapsedTime.append(time.time()-self.StartT)
                env.runningRewards = 0
                env.runningLength = 0
            obsL.append(obs)
            rewardL.append(info['NReward'] if env.NORMALIZED else reward)
            doneL.append(done)
            infoL.append(info)
        return np.array(obsL).reshape(-1,self.observation_space.shape[0]), rewardL, doneL, infoL

    def render(self):
        self.envL[0].render()

class Normalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        gym.Wrapper.__init__(self, env)
        self.ob_rms = RunningMeanStd(shape=(1,self.observation_space.shape[0])) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            nRews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            infos['NReward']=nRews
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        obs=obs.reshape((1,-1))
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(1)
        obs = self.env.reset()
        return self._obfilt(obs)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
