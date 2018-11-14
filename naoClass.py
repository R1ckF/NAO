import naoqi
import almath
import sys
import time
import motion
import qi
from gym import spaces
import numpy as np
IP = "169.254.36.168"
# IP = "127.0.0.1"
PORT = 9559
SPEED = 0.1


class NaoMotion:

    def __init__(self, IP, PORT, SPEED):

        self.session = qi.Session()
        self.session.connect("tcp://"+IP+":"+str(PORT))
        self.motionService = self.session.service("ALMotion")
        self.memP = naoqi.ALProxy("ALMemory",IP,PORT)
        self.speed = SPEED
        self.bodyNames = self.motionService.getBodyNames("Body")
        self.high = np.array([2.0857, 0.5149, 2.0857, 1.3265, 2.0857, -0.0349,\
         1.8238, 1, 0.740810, 0.790477, 0.484090, 2.112528, 0.922747, 0.769001, \
         0.740810, 0.379472, 0.484090, 2.120198, 0.932056, 0.397935, 2.0857,  0.3142,\
          2.0857, 1.5446, 1.8238, 1])
        self.low = np.array([-2.0857, -0.6720, -2.0857, -0.3142, -2.0857, -1.5446,\
         -1.8238, 0, -1.145303, -0.379472, -1.535889, -0.092346, -1.189516, -0.397880,\
          -1.145303, -0.790477, -1.535889, -0.103083, -1.186448, -0.768992, -2.0857,\
           -1.3265, -2.0857, 0.0349, -1.8238, 0])
        self.sensorNames = ['LFsrFL', 'LFsrFR', 'LFsrBL','LFsrBR', 'RFsrFL','RFsrFR','RFsrBL','RFsrBR', 'GyrX', 'GyrY', 'AccX', 'AccY', 'AccZ', 'TorsoAngleX', 'TorsoAngleY']
        self.time = time.time()
        self.pathNames = ["Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value",
        "Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value",
        "Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value",
        "Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value",
        "Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value",
        "Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value",
        "Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value",
        "Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/GyrX/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/GyrY/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/AccX/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/AccY/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/AccZ/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value",
        "Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value"]
        self.observation_space = len(self.bodyNames+self.sensorNames)
        self.action_space = spaces.Box(low=self.low, high=self.high)
        self.COM = self.motionService.getCOM
        self.sPoly = self.motionService.getSupportPolygon

	def setAll(self,angles):
		return self.motionService.setAngles(self.bodyNames, angles, self.speed)

	def getAll(self):
		return self.motionService.getAngles(self.bodyNames,True), [self.memP.getData(path) for path in self.pathNames] #True forces use of sensors

    def step(self,action):
        print(action.shape, action)
        action = np.clip(action, self.low, self.high)
        self.setAll(action)
        angles, sensors = self.getAll()
        return angles + sensors, 1 + sensors[10], False, None # return observation, rewards signal (acceleration forward sensors[10] verify), dones, info

    def reset(self):
        self.motionService.setFallManagerEnabled(False)
        self.motionService.wakeUp()
        angles, sensors = self.getAll()
        return angles+sensors

    def rest(self):
        self.motionService.rest()


    def setAll(self,angles):
        # print(angles.shape, len(list(angles)))
        return self.motionService.setAngles(self.bodyNames, angles.tolist(), self.speed)

    def getAll(self):
        return self.motionService.getAngles(self.bodyNames,True), [self.memP.getData(path) for path in self.pathNames] #True forces use of sensors

naom = NaoMotion(IP, PORT, SPEED)


class Normalize:
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, nao, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.nao = nao
        self.ob_rms = RunningMeanStd(shape=(1,self.nao.observation_space)) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nao.observation_space,))
        self.action_space = self.nao.action_space
    def step(self, action):
        obs, rews, news, infos = self.nao.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(np.array(obs))
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            nRews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos, nRews

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
        obs = self.nao.reset()
        return self._obfilt(np.array(obs))

    def close(self):
        self.nao.rest()

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
