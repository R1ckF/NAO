import naoqi as nao
import almath
import sys
import time
import motion
import qi

# IP = "169.254.40.45"
IP = "127.0.0.1"
PORT = 9559
SPEED = 0.1

# nao
# nlr123
# tts = nao.ALProxy("ALTextToSpeech", IP, PORT)

# class jointSensors(object):
#
# 	def __init__(self,joint):
#
# 		self.joint = joint
#
# 	@property
# 	def value(self):
# 		return motionService.getAngles(self.joint,True)[0]
#
# 	@value.setter
# 	def value(self, val):
# 		motionService.setAngles(self.joint, val, speed)
# 		self._value = motionService.getAngles(self.joint,False)
#
#
# class sensors(object):
#
# 	def __init__(self,sensor,path):
#
# 		self.sensor = sensor
# 		self.path = path
#
# 	@property
# 	def value(self):
# 		return memP.getData(self.path)
#
# 	@value.setter
# 	def value(self, value):
# 		print("read only sensor, can not set")
# 		return None
class NaoMotion:

	def __init__(self, IP, PORT, SPEED):

		self.session = qi.Session()
		self.session.connect("tcp://"+IP+":"+str(PORT))
		self.motionService = self.session.service("ALMotion")
		self.memP = nao.ALProxy("ALMemory",IP,PORT)
		self.speed = SPEED
		self.bodyNames = self.motionService.getBodyNames("Body")
		self.sensorNames = ['LFsrFL', 'LFsrFR', 'LFsrBL','LFsrBR', 'RFsrFL','RFsrFR','RFsrBL','RFsrBR', 'GyrX', 'GyrY', 'AccX', 'AccY', 'AccZ', 'TorsoAngleX', 'TorsoAngleY']

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


	def setAll(self,angles):
		return self.motionService.setAngles(self.bodyNames, angles, self.speed)

	def getAll(self):
		return self.motionService.getAngles(self.bodyNames,True)#, [self.memP.getData(path) for path in self.pathNames] #True forces use of sensors


naom = NaoMotion(IP, PORT, SPEED)
stance = naom.getAll()
print(stance)
# print(b)

naom.setAll([0 for i in range(len(stance))])
time.sleep(1)
# naom.motionService.waitUntilMoveIsFinished()
naom.setAll([stanc+0.1 for stanc in stance])
