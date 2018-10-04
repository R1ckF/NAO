import naoqi as nao
import almath
import sys
import time
import motion
import qi

# IP = "192.168.43.99"
IP = "169.254.132.169"
PORT = 9559
speed = 0.5

# nao
# nlr123
tts = nao.ALProxy("ALTextToSpeech", IP, PORT)

class jointSensors(object):

	def __init__(self,joint):
	
		self.joint = joint
	
	@property
	def value(self):
		return motionService.getAngles(self.joint,True)[0]

	@value.setter
	def value(self, val):
		motionService.setAngles(self.joint, val, speed)
		self._value = motionService.getAngles(self.joint,False)


class sensors(object):

	def __init__(self,sensor,path):
		
		self.sensor = sensor
		self.path = path
	
	@property
	def value(self):
		return memP.getData(self.path)

	@value.setter
	def value(self, value):
		print("read only sensor, can not set")
		return None

session = qi.Session()
session.connect("tcp://"+IP+":"+str(PORT))
motionService = session.service("ALMotion")
memP = nao.ALProxy("ALMemory",IP,PORT)

def setAll(angles, speed=speed):
	motionService.setAngles(bodyNames, angles, speed)

def getAll():
	return motionService.getAngles(bodyNames,True), [memP.getData(path) for path in pathNames]

bodyNames = motionService.getBodyNames("Body")
sensorNames = ['LFsrFL', 'LFsrFR', 'LFsrBL','LFsrBR', 'RFsrFL','RFsrFR','RFsrBL','RFsrBR', 'GyrX', 'GyrY', 'AccX', 'AccY', 'AccZ', 'TorsoAngleX', 'TorsoAngleY']

pathNames = ["Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value",
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

sensorD = {}
for item in bodyNames:
	sensorD[item] = jointSensors(item)
for (item, itempath) in zip(sensorNames,pathNames):
	sensorD[item] = sensors(item, itempath)

print(bodyNames, sensorNames)



motionService.wakeUp()
# setAngles(motionService, 'RShoulderPitch', -0.7, speed)
# setAngles(motionService, 'RElbowRoll', 0, speed)
# setAngles(motionService, 'RElbowYaw', 0, speed)
# setAngles(motionService, 'RHand', 1,speed)
# setAngles(motionService, 'LShoulderPitch', -0.7, speed)
# setAngles(motionService, 'LElbowRoll', 0, speed)
# setAngles(motionService, 'LElbowYaw', 0, speed)
# setAngles(motionService, 'LHand', 1,speed)
# time.sleep(1.5)
# setAngles(motionService, 'RShoulderPitch', 1, speed)
# setAngles(motionService, 'RElbowRoll', 0, speed)
# setAngles(motionService, 'RElbowYaw', 0, speed)
# setAngles(motionService, 'RHand', 1,speed)
# setAngles(motionService, 'LShoulderPitch', 1, speed)
# setAngles(motionService, 'LElbowRoll', 0, speed)
# setAngles(motionService, 'LElbowYaw', 0, speed)
# setAngles(motionService, 'LHand', 1,speed)
# time.sleep(1.5)
# motionService.rest()

# sensorAngles = getSensors(memP,motionService)


# for i in range(len(bodyNames)):
# 	print sensorNames[i], sensorAngles[i]
# 	tts.say("Changing "+sensorNames[i])
# 	time.sleep(0.5)
# 	setAngles(motionService, sensorNames[i], sensorAngles[i]+0.5, 0.1)
# 	time.sleep(2)
# 	sensorAngles = getSensors(memP,motionService)
# 	print sensorNames[i], sensorAngles[i]
# 	setAngles(motionService, sensorNames[i], sensorAngles[i]-0.5, 0.1)
# 	time.sleep(2)
# 	sensorAngles = getSensors(memP,motionService)
# 	print sensorNames[i], sensorAngles[i]
# motionService.rest()
# if __name__=='__main__':
# 	main()