
"""
AUTHORS:
    Ernest J. Quant
    Ivan A. Reyes
    Orson Meyreles
    John Quitto-Graham
    Carlos Valdes
    Maria Celeste Carbonell
"""
from mpl_toolkits.mplot3d import Axes3D
from queue import PriorityQueue
from airsim import *
import matplotlib.pyplot as plt
import setup_path
import airsim
import math
import time
import os


class Node:

    def __init__ (self,cargo=None,parent=None):
        self.cargo = cargo
        self.parent = parent

    def __str__(self):
        return str(self.cargo)

    def getCargo(self):
        return self.cargo

    def getParent(self):
        return self.parent


def printPathAndGraph(lastNode):

    path = []
    xCords = []
    yCords = []
    zCords = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while lastNode.parent != None:
        x,y = lastNode.getCargo()
        xCords.insert(0,x)
        yCords.insert(0,y)
        zCords.insert(0,-1)
        path.insert(0,(x,y))
        lastNode = lastNode.parent

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    print(path)


def graphPath():
    
    return 0


def calcHeuristic(startX,startY,goalX,goalY):
    heuristic = math.sqrt(((goalX-startX)**2+(goalY-startY)**2))

    return heuristic


"""
Checks to see if the drone can move front, left, right or backwards from it's current position.
"""
def checkAdj(xCord,yCord,visitedList,drone):
    north = False
    east = False
    south = False
    west = False
    noSafe = False

    tmp_dir = r"C:\Users\Ernest\Pictures\DronePic"
    imagequeue = []
    direct = [north,east,south,west]


    # FRONT PICTURE
    drone.moveToPositionAsync(3,0,-1,0.00001,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
    time.sleep(5)
    responses = drone.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
    imagequeue.append(responses[0].image_data_uint8)

    # RIGHT PICTURE
    drone.moveToPositionAsync(0,3,-1,0.00001,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
    time.sleep(5)
    responses = drone.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
    imagequeue.append(responses[0].image_data_uint8)

    # BACK PICTURE
    drone.moveToPositionAsync(-3,0,-1,0.00001,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
    time.sleep(5)
    responses = drone.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
    imagequeue.append(responses[0].image_data_uint8)

    # LEFT PICTURE
    drone.moveToPositionAsync(0,-3,-1,0.00001,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
    time.sleep(5)
    responses = drone.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
    imagequeue.append(responses[0].image_data_uint8)

    # FACE FRONT AGAIN
    drone.moveToPositionAsync(3,0,-1,0.00001,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))


    for i in range(len(imagequeue)):
        filename = os.path.join(tmp_dir, str(i))
        airsim.write_file(os.path.normpath(filename + '.png'),imagequeue[i])
    

    """ USED FOR TESTING, DELETE WHEN FINISHED """
    yah = input("n true: ").lower()
    if yah == "t":
        north = True

    yah = input("e true: ").lower()
    if yah == "t":
        east = True

    yah = input("s true: ").lower()
    if yah == "t":
        south = True

    yah = input("w true: ").lower()
    if yah == "t":
        west = True

    """ WILL BE ACTUAL IMPLEMENTATION WHEN CNN IS FINISHED, DELETE ABOVE WHEN UNCOMMENTING THIS"""
    """if safeCheck() and not in visited:
        north = True
    if safeCheck() and not in visited:
        east = True
    if safeCheck() and not in visited:
        south = True
    if safeCheck() and not in visited:
        west = True"""

    if not any(direct):
        noSafe = True

    return north, east, south, west, noSafe


def yahSearch(goal):

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    goalX,goalY = goal
    isGoal = False
    pq = PriorityQueue()

    visited = []
    currentNode = Node((0,0))
    isGoal = False
    pq.put((1,(currentNode)))

    while not isGoal:
        z = pq.get()
        while not pq.empty():
            pq.get()

        
        h = z[0]            
        x,y = z[1].getCargo()
        print(x,y)
        currentNode = z[1]
        print(currentNode)
        client.moveToPositionAsync(x,y,-1,5,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
        visited.append((x,y))

        # GOAL TEST
        if (x,y) == (goalX,goalY):
            break

        northSafe, eastSafe, southSafe, westSafe,noneSafe = checkAdj(x,y,visited,client)

        if northSafe:
            newNode1 = Node((x+10,y),currentNode)
            dist = calcHeuristic(x+10,y,goalX,goalY)
            pq.put((dist,(newNode1)))
        if eastSafe: 
            newNode2 = Node((x,y+10),currentNode)
            dist = calcHeuristic(x,y+10,goalX,goalY)
            pq.put((dist,(newNode2)))
        if southSafe:
            newNode3 = Node((x-10,y),currentNode)
            dist = calcHeuristic(x-10,y,goalX,goalY)
            pq.put((dist,(newNode3)))
        if westSafe:
            newNode4 = Node((x,y-10),currentNode)
            dist = calcHeuristic(x,y-10,goalX,goalY)
            pq.put((dist,(newNode4)))
        if noneSafe:
            currentNode = currentNode.parent
            x,y = currentNode.getCargo()
            h = calcHeuristic(x,y,goalX,goalY)
            pq.put((h,(currentNode)))

    returnPath(currentNode)


def main():
    
    yahSearch((60,100))

if __name__ == "__main__":
    main()