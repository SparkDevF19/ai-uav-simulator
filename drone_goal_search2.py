"""
AUTHORS:
    Ernest J. Quant
    Ivan A. Reyes
    Orson Meyreles
    John Quitto-Graham
    Carlos Valdes
    Maria Celeste Carbonell
"""

from numpy import loadtxt
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from mpl_toolkits.mplot3d import Axes3D
from queue import PriorityQueue
import setup_path
import airsim
from airsim import *
import matplotlib.pyplot as plt
import math
import time
import os

#Will be used to return the successful path
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

    ax.scatter(xCords, yCords, zCords, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    print(path)


#Checks current path in the X,Y coordinate direction for possible collision
def checkSafe(drone,yawVal,droneNetwork):
    
    imagequeue = []
    tmp_dir = r"C:\Users\Ernest\Pictures\DronePic"

    drone.moveByAngleZAsync(0,0,-1,yaw=yawVal,duration=1)
    time.sleep(4)
    responses = drone.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
    imagequeue.append(responses[0].image_data_uint8)

    filename = os.path.join(tmp_dir, "current")
    airsim.write_file(os.path.normpath(filename + '.png'),imagequeue[0])

    img_addr = r"C:\Users\Ernest\Pictures\DronePic\current.png"

    img = image.load_img(img_addr, target_size = (80,80))
    img_test = image.img_to_array(img)
    img_test = np.expand_dims(img_test, axis=0)
    img_test = preprocess_input(img_test)

    result = droneNetwork.predict(img_test)

    if result == [[1.]]:
        return True
    else:
        return False


def calcHeuristic(startX,startY,goalX,goalY):
    heuristic = math.sqrt(((goalX-startX)**2+(goalY-startY)**2))

    return heuristic


def yahSearch(goal,droneNetwork):
    moveMag = 10
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
    hpriq = PriorityQueue()

    inInitial = True

    while not isGoal:

        
        time.sleep(3)
        z = pq.get()
        h = z[0]
        x,y = z[1].getCargo()
        curX, curY = (x,y)
        currentNode = z[1]
        
        time.sleep(3)
        visited.append((x,y))

        # GOAL TEST
        if (x,y) == (goalX,goalY):
            print("GOAL REACHED!")
            break

        if (x+moveMag,y) not in visited:
            upH = (calcHeuristic(x+moveMag,y,goalX,goalY),(x+moveMag,y),0)
            hpriq.put(upH)

        if (x,y-moveMag) not in visited:
            leftH = (calcHeuristic(x,y-moveMag,goalX,goalY),(x,y-moveMag),-1.5)
            hpriq.put(leftH)
         
        if inInitial:
            downH = (calcHeuristic(x-moveMag,y,goalX,goalY),(x-moveMag,y),3)
            hpriq.put(downH)
            inInitial = False
        
        if (x,y+moveMag) not in visited:
            rightH = (calcHeuristic(x,y+moveMag,goalX,goalY),(x,y+moveMag),1.5)
            hpriq.put(rightH)

        safeMove = False
        
        while not hpriq.empty():
            movement = hpriq.get()
            x, y = movement[1]
            h = movement[0]
            
            if checkSafe(client,movement[2],droneNetwork):
                time.sleep(5)

                safeMove = True
                newNode = Node((x,y),currentNode)

                pq.put((h,(newNode)))
                while not hpriq.empty():
                    hpriq.get()

                client.moveToPositionAsync(x,y,-1,3,drivetrain=DrivetrainType.ForwardOnly,yaw_mode=YawMode(False,0))
                time.sleep(7)
                client.hoverAsync()
                break
        
        if safeMove == False:

            currentNode = currentNode.parent
            x,y = currentNode.getCargo()
            h = calcHeuristic(x,y,goalX,goalY)
            pq.put((h,(currentNode)))

    printPathAndGraph(currentNode)


def main():
    
    network = tf.keras.models.load_model(r'C:\Users\Ernest\AirSim\PythonClient\multirotor\CNN_tester.model')
    yahSearch((100,20),network)

if __name__ == "__main__":
    main()