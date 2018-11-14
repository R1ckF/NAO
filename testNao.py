import naoqi as nao
import almath
import sys
import time
import motion
import qi

# IP = "192.168.43.99"
IP = "169.254.37.89"
# IP = "127.0.0.1"
PORT = 9559
speed = 0.5

# nao
# nlr123

def getSensors(memP, motionService):


	LFsrFL = memP.getData("Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value")
	LFsrFR = memP.getData("Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value")
	LFsrBL = memP.getData("Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value")
	LFsrBR = memP.getData("Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value")

	# print("Left FSR [kg]: % .2f %.2f %.2f %.2f" % (LFsrFL, LFsrFR, LFsrBL, LFsrBR))
	# print("Total Left [kg]: ", sum([LFsrFL, LFsrFR, LFsrBL, LFsrBR]))

	RFsrFL = memP.getData("Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value")
	RFsrFR = memP.getData("Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value")
	RFsrBL = memP.getData("Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value")
	RFsrBR = memP.getData("Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value")

	GyrX = memP.getData("Device/SubDeviceList/InertialSensor/GyrX/Sensor/Value")
	GyrY = memP.getData("Device/SubDeviceList/InertialSensor/GyrY/Sensor/Value")

	AccX = memP.getData("Device/SubDeviceList/InertialSensor/AccX/Sensor/Value")
	AccY = memP.getData("Device/SubDeviceList/InertialSensor/AccY/Sensor/Value")
	AccZ = memP.getData("Device/SubDeviceList/InertialSensor/AccZ/Sensor/Value")

	TorsoAngleX = memP.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
	TorsoAngleY = memP.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")

	bodyNames = motionService.getBodyNames("Body")
	bodyNames = bodyNames + ['LFsrFL', 'LFsrFR', 'LFsrBL','LFsrBR', 'RFsrFL','RFsrFR','RFsrBL','RFsrBR', 'GyrX', 'GyrY', 'AccX', 'AccY', 'AccZ', 'TorsoAngleX', 'TorsoAngleY']
	sensorAngles = motionService.getAngles("Body",True)
	sensorAngles = sensorAngles + [LFsrFL, LFsrFR, LFsrBL,LFsrBR, RFsrFL,RFsrFR,RFsrBL,RFsrBR, GyrX, GyrY, AccX, AccY, AccZ, TorsoAngleX, TorsoAngleY]

	# sensorAngles = motionService.getAngles("Body",True)
	return bodyNames, sensorAngles

tts = nao.ALProxy("ALTextToSpeech", IP, PORT)

# tts.say("I only serve my master, Rick ")
# tts.say("Can I get you something to drink?")
#
memP = nao.ALProxy("ALMemory",IP,PORT)

# def ForceSensor():
# 	LFsrFL = memP.getData("Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value")
# 	LFsrFR = memP.getData("Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value")
# 	LFsrBL = memP.getData("Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value")
# 	LFsrBR = memP.getData("Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value")

# 	# print("Left FSR [kg]: % .2f %.2f %.2f %.2f" % (LFsrFL, LFsrFR, LFsrBL, LFsrBR))
# 	# print("Total Left [kg]: ", sum([LFsrFL, LFsrFR, LFsrBL, LFsrBR]))

# 	RFsrFL = memP.getData("Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value")
# 	RFsrFR = memP.getData("Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value")
# 	RFsrBL = memP.getData("Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value")
# 	RFsrBR = memP.getData("Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value")

# 	# print("Right FSR [kg]: % .2f %.2f %.2f %.2f" % (RFsrFL, RFsrFR, RFsrBL, RFsrBR))
# 	# print("Total Right [kg]: ", sum([RFsrFL, RFsrFR, RFsrBL, RFsrBR]))

# 	# print("Total Weight [kg]: ", sum([LFsrFL,
# 	# 	LFsrFR, LFsrBL, LFsrBR,RFsrFL, RFsrFR, RFsrBL, RFsrBR]))

# 	return LFsrFL, LFsrFR, LFsrBL, LFsrBR, RFsrFL, RFsrFR, RFsrBL, RFsrBR

# def IMU():
# 	GyrX = memP.getData("Device/SubDeviceList/InertialSensor/GyrX/Sensor/Value")
# 	GyrY = memP.getData("Device/SubDeviceList/InertialSensor/GyrY/Sensor/Value")

# 	AccX = memP.getData("Device/SubDeviceList/InertialSensor/AccX/Sensor/Value")
# 	AccY = memP.getData("Device/SubDeviceList/InertialSensor/AccY/Sensor/Value")
# 	AccZ = memP.getData("Device/SubDeviceList/InertialSensor/AccZ/Sensor/Value")

# 	TorsoAngleX = memP.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
# 	TorsoAngleY = memP.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")

# 	# print("Gyrometers Value X: %.3f, Y: %.3f" % (GyrX, GyrY))
# 	# print("Accelerometers value X: %.3f, Y: %.3f, Z: %.3f" %(AccX, AccY, AccZ))
# 	# print("Torso angles [radian] X: %.3f, Y: %.3f" %(TorsoAngleX, TorsoAngleY))

# 	return GyrX, GyrY, AccX, AccY, AccZ, TorsoAngleX, TorsoAngleY


# for i in range(10):
# 	# ForceSensor()
# 	IMU()
# 	time.sleep(5)

# def StiffnessOn(proxy):
# 	pNames = "Body"
# 	pStiffnessLists = 1.0
# 	pTimeLists = 1.0
# 	proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

# def main():
# 	try:
# 		motionP = nao.ALProxy("ALMotion", IP, PORT)
# 	except Exception, e:
# 		print "Error: ", e

# 	try:
# 		postureP = nao.ALProxy("ALRobotPosture", IP, PORT)
# 	except Exception, e:
# 		print "Error: ", e

# 	StiffnessOn(motionP)

# 	postureP.goToPosture("StandInit", speed)
# 	# postureP.goToPosture("StandZero",speed)
# 	postureP.goToPosture("Crouch",speed)
# main()

def main(session):
	motionService = session.service("ALMotion")

	# useSensors = False
	# commandAngles = motionService.getAngles("Body",useSensors)
	# print(commandAngles)

	# sensorAngles = motionService.getAngles("Body",True)

	# for sensor in sensorAngles:
	# 	print(sensor)

	# print("Error")
	# for i in range(len(commandAngles)):
	# 	print(commandAngles[i]-sensorAngles[i])

	# print(len(commandAngles))

	postureService = session.service("ALRobotPosture")
	# bodyNames = motionService.getBodyNames("Body")
	# bodyNames = bodyNames + ['LFsrFL', 'LFsrFR', 'LFsrBL','LFsrBR', 'RFsrFL','RFsrFR','RFsrBL','RFsrBR', 'GyrX', 'GyrY', 'AccX', 'AccY', 'AccZ', 'TorsoAngleX', 'TorsoAngleY']
	# sensorAngles = motionService.getAngles("Body",True)
	# sensorL = sensorDict()
	bodyNames, sensorAngles = getSensors(memP,motionService)
	# sensorAngles = sensorAngles + sensorL
	print(len(sensorAngles))
	print(len(bodyNames))
	# for sensor in sensorList:
	# 	print sensor
	for i in range(len(bodyNames)):
		print bodyNames[i], sensorAngles[i]
	# print ""
	# print motionService.getSummary()
	# motionService.wakeUp()
	# tts.say("I am tired, I am going to sit down again")
	postureService.goToPosture("StandZero",speed)
	# tts.say("I only serve my master, Rick ")
	for i in range(10):
		time.sleep(0.5)
		postureService.goToPosture("Stand",speed)
		time.sleep(0.5)
		postureService.goToPosture("StandZero",speed)
	postureService.goToPosture("LyingBelly",speed)

	time.sleep(10)
	motionService.rest()
session = qi.Session()

session.connect("tcp://"+IP+":"+str(PORT))
main(session)
