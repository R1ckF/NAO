# import vidcap
import qi
import argparse
import sys
import time
import pygame
from rayCasting import interpolate, insidePolygon, scalePolygon
import numpy as np
import naoqi
# display settings
displayScale = 1000 # pixels/m
screenWidth = 700   # pixels
screenHeight = 700  # pixels

displayScale = 1000 # pixels/m
screenWidth = 700   # pixels
screenHeight = 700  # pixels
screenColor = pygame.Color("white")
polygonColor = pygame.Color("red")
scaledpolygonColor = pygame.Color("blue")
circleColorInside = pygame.Color("green")
circleColorOutside = pygame.Color("blue")

screenColor = pygame.Color("white")
polygonCommandColor = pygame.Color("green")
polygonSensorColor = pygame.Color("red")
comColor = pygame.Color("blue")

def almotionToPygame(point):
    """
    Links ALMotion to Pygame.
    """
    return [int(-point[1] * displayScale + 0.5 * screenWidth),
            int(-point[0] * -displayScale + 0.5 * screenHeight)]

def drawPolygon(screen, data, color) :
    """
    Draw a Polygon.
    """
    if (data != []):
        # print(map(almotionToPygame, data))
        pygame.draw.lines(screen, color, True, map(almotionToPygame, data), 1)

def drawPoint(screen, point, color) :
    """
    Draw a Point.
    """
    pygame.draw.circle(screen, color, almotionToPygame(point), 3)

def main(session, frame) :
    """
    This example uses the getSupportPolygon method.
    """
    # Get the service ALMotion.

    motion_service  = session.service("ALMotion")
    mem = naoqi.ALProxy("ALMemory",args.ip,args.port)
    paths = ["Device/SubDeviceList/InertialSensor/AccX/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AccY/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/AccZ/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/GyrX/Sensor/Value",
    "Device/SubDeviceList/InertialSensor/GyrY/Sensor/Value"]

    def accs():
        return [mem.getData(path) for path in paths]

    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Robot Support Polygon")
    frameN = 0
    running = True
    while (running):
        # Check if user has clicked 'close'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(screenColor)

        supportPolygonCommand = np.array(motion_service.getSupportPolygon(frame, True))
        mean = np.mean(supportPolygonCommand,axis=0)
        supportPolygonCommand = supportPolygonCommand-mean

        scaledPolygon = scalePolygon(supportPolygonCommand, 0.78)

        drawPolygon(screen, supportPolygonCommand, polygonCommandColor)

        supportPolygonSensor = motion_service.getSupportPolygon(frame, True)
        drawPolygon(screen, scaledPolygon, polygonSensorColor)

        com = motion_service.getCOM("Body", frame, False)[0:2]-mean
        accxyz = accs()
        # drawPoint(screen, com, comColor)

        if insidePolygon(com, scaledPolygon):
            drawPoint(screen, com, circleColorInside)

        else:
            drawPoint(screen, com, circleColorOutside)
            # motion_service.wakeUp()
        accX, accY, accZ, GyrX, GyrY = accxyz[0],accxyz[1],accxyz[2], accxyz[3],accxyz[4]
        pygame.draw.line(screen, comColor, almotionToPygame(com), almotionToPygame([com[0]+accX, com[1]+accY]), 1)
        if frameN%50 ==0: print(accX, accY, accZ, GyrX, GyrY)
        pygame.display.flip()
        time.sleep(0.02)
        frameN+=1
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    parser.add_argument("--frame", type=str, choices =['world', 'robot'], default = 'world')

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    if (args.frame == 'world'):
        frame = 1
    else:
        frame = 2
    main(session, frame)
