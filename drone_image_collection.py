"""
IMAGE COLLECTION
--------------------------------------------------------------------------------------------------------------
ADAPTED FROM: 
    Availability: https://github.com/simondlevy/AirSimTensorFlow/blob/master/image_collection.py
    Title: image_collection.py
    Authors: Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, 
    Simon D. Levy, Will McMurtry, Jacob Rosen
CHANGES FROM ABOVE: 
    Changed airsim client from CarClient to MultirotorClient, and also took out enableAPIControl to be able
    to gather the collision images from collisions made by piloting the multirotor manually, using a RC.
    Instead of terminating after a collision, the code sleeps to allow the pilot to move away from the obstacle 
    and the next set of images are collected upon the next collision, and so on.
CHANGES MADE BY: 
    Ernest Quant

DRONE 360-EXPLORATION
--------------------------------------------------------------------------------------------------------------
DESCRIPTION:
    Added continuous collection of training images by allowing the drone to explore a linear path from origin 
    to the boundaries of the environment. Each exploration path is rotated 5 degrees (counterclockwise). 
    Increased QUEUESIZE to 100 to allow the collection of more imges prior to a collition, including SAFE 
    situations. 
CHANGES MADE BY: 
    Ivan A. Reyes
"""
import airsim
import pprint
import time
import os
import math



client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# CHANGE TO YOUR DESIRED DIRECTORY
tmp_dir = r"C:\Users\Ernest\Pictures\Collision_Images"

try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise


imagequeue = []
QUEUESIZE = 200 # NUMBER OF IMAGES KEPT RIGHT BEFORE AND UP TO A COLLISION
imageBatch = 0  # USED TO INCREMENT FILE NUMBERS AFTER COLLISION
#endNum = 100   # IF YOU WANT TO TERMINATE AT A CERTAIN NUMBER OF IMAGES

degrees = list(range(0,360,5)) #from 0 to 360 in intervals of 5
travel_distance = 50 
airsim.wait_key('Press any key to take off and start 360-exploration starting from Origin (height = 20m) at 15 m/s')

while True:

    for degree in degrees:
        #Automates the position and data collection
        client.takeoffAsync().join()
        xcoor = travel_distance * math.cos(math.radians(degree))
        ycoor = travel_distance * math.sin(math.radians(degree))
        client.moveToPositionAsync(xcoor, ycoor, -20  , 15)
        client.hoverAsync().join()
        
        while True:

            responses = client.simGetImages([airsim.ImageRequest(1,airsim.ImageType.Scene)])

            imagequeue.append(responses[0].image_data_uint8)

            # KEEPS THE IMAGE QUEUE POPULATED WITH THE MOST RECENT IMAGES
            if len(imagequeue) == QUEUESIZE:
                for i in range(QUEUESIZE):
                    filename = os.path.join(tmp_dir, str(i + imageBatch))
                    airsim.write_file(os.path.normpath(filename + '.png'),imagequeue[i])
                imagequeue.pop(0)

            collision_info = client.simGetCollisionInfo()
            # PRINTS COLLISION INFORMATION AFTER COLLISION AND THEN BREAKS
            if collision_info.has_collided:
                print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
                    pprint.pformat(collision_info.position), 
                    pprint.pformat(collision_info.normal), 
                    pprint.pformat(collision_info.impact_point), 
                    collision_info.penetration_depth, collision_info.object_name, collision_info.object_id))
                break

            time.sleep(0.1)

        # UNCOMMENT TO TERMINATE AFTER A SET NUMBER OF IMAGES
        """if imageBatch > endNum:
            break"""

        imageBatch += QUEUESIZE
        client.reset()

client.reset()