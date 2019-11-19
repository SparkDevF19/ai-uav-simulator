"""
IMAGE COLLECTION & DRONE 360-EXPLORATION
--------------------------------------------------------------------------------------------------------------
DESCRIPTION:
    Changed airsim client from CarClient to MultirotorClient, and also took out enableAPIControl to be able
    to gather the collision images from collisions made by piloting the multirotor manually, using a RC.
    Instead of terminating after a collision, the code sleeps to allow the pilot to move away from the obstacle 
    and the next set of images are collected upon the next collision, and so on.
    
    Added continuous collection of training images by allowing the drone to explore a linear path from origin 
    to the boundaries of the environment. Each exploration path is rotated 5 degrees (counterclockwise). 
    Increased QUEUESIZE to 200 to allow the collection of more imges prior to a collition, including SAFE 
    situations. Added a timer to proceed to next path after enough time in collition-free trayectory.
--------------------------------------------------------------------------------------------------------------    
MADE BY: 
    Ivan A. Reyes
    Ernest J. Quant
    Orson Meyreles
    John Quitto-Graham  
    Catherine Angelini
    Carlos Valdes
--------------------------------------------------------------------------------------------------------------

"""
import setup_path
import airsim
import pprint
import shutil
import time
import os
import math

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# CHANGE TO YOUR DESIRED DIRECTORY
safe_dirPFM = "D:\\Airsim_SafePFM"
unsafe_dirPFM = "D:\\Airsim_UnsafePFM"
safe_dirPNG = "D:\\Airsim_SafePNG"
unsafe_dirPNG = "D:\\Airsim_UnsafePNG"

try:
    os.makedirs(safe_dirPFM)
    os.makedirs(safe_dirPNG)
    os.makedirs(unsafe_dirPFM)
    os.makedirs(unsafe_dirPNG)
except OSError:
    if not os.path.isdir(safe_dirPFM):
        raise
    if not os.path.isdir(safe_dirPNG):
        raise
    if not os.path.isdir(unsafe_dirPFM):
        raise
    if not os.path.isdir(unsafe_dirPNG):
        raise


imagequeue = []
QUEUESIZE = 200 # NUMBER OF IMAGES KEPT RIGHT BEFORE AND UP TO A COLLISION
isPFM = False
isPNG = False
imageBatch = 0  # USED TO INCREMENT FILE NUMBERS AFTER COLLISION
#endNum = 100   # IF YOU WANT TO TERMINATE AT A CERTAIN NUMBER OF IMAGES

degrees = list(range(0,360,5)) #from 0 to 360 in intervals of 5
velocityMagnitude = 5

yah = input("T for PFM, F for PNG: ").lower()
if yah == "t":
    isPFM = True
    fileExt = "pfm"
else:
    isPNG = True
    fileExt = "png"

airsim.wait_key('Press any key to take off and start 360-exploration starting from Origin (height = 20m) at 15 m/s')

while True:
    #Automates the position and data collection
    
    for degree in degrees:
        
        print("Preparing..", end='.')
        client.armDisarm(True)#ADDED NOW
        client.enableApiControl(True)
        client.takeoffAsync().join()

        print("Ready!")
        timeout = 15   #in seconds
        timeout_start = time.time()

        vx0 =  math.cos(math.radians(degree))
        vy0 = math.sin(math.radians(degree))
        vx = velocityMagnitude * vx0
        vy = velocityMagnitude * vy0

        client.moveByVelocityAsync(vx, vy, 0, 20)
        
        while time.time() < timeout_start + timeout:
            
            if isPNG:
                responses = client.simGetImages([airsim.ImageRequest(0,airsim.ImageType.Scene)])
                imagequeue.append(responses[0].image_data_uint8)
            elif isPFM:
                responses = client.simGetImages([airsim.ImageRequest(0,airsim.ImageType.DepthVis,True,False)])
                imagequeue.append(airsim.get_pfm_array(responses[0]))

            # KEEPS THE IMAGE QUEUE POPULATED WITH THE MOST RECENT IMAGES
            if len(imagequeue) == QUEUESIZE:
                for i in range(QUEUESIZE):
                    # SAVE TO CORRECT SAFE DIRECTORY BASED ON PFM/PNG
                    if isPNG:
                        filename = os.path.join(safe_dirPNG, str(i + imageBatch))
                        airsim.write_file(os.path.normpath(filename + '.png'),imagequeue[i])
                    elif isPFM:
                        filename = os.path.join(safe_dirPFM, str(i + imageBatch))
                        airsim.write_pfm(os.path.normpath(filename + '.pfm'),imagequeue[i])
                imagequeue.pop(0)

            collision_info = client.simGetCollisionInfo()
            # PRINTS COLLISION INFORMATION AFTER COLLISION AND THEN BREAKS
            if collision_info.has_collided:
                print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
                    pprint.pformat(collision_info.position), 
                    pprint.pformat(collision_info.normal), 
                    pprint.pformat(collision_info.impact_point), 
                    collision_info.penetration_depth, collision_info.object_name, collision_info.object_id))

                # ONCE COLLISION OCCURS MOVE THE LAST 15 IMAGES TO THE CORRECT UNSAFE DIRECTORY
                for i in range((imageBatch+QUEUESIZE-15),imageBatch+QUEUESIZE):
                   # SAVE TO CORRECT UNSAFE DIRECTORY BASED ON PFM/PNG
                    if isPNG:
                        shutil.move(safe_dirPNG + "\\{}.{}".format(i,fileExt),unsafe_dirPNG)
                    elif isPFM:
                        shutil.move(safe_dirPFM + "\\{}.{}".format(i,fileExt),unsafe_dirPFM)
                break

        print("Disarming...")
        client.armDisarm(False)
            
        # UNCOMMENT TO TERMINATE AFTER A SET NUMBER OF IMAGES
        """if imageBatch > endNum:
            break"""
        
        print("Restarting!")
        imageBatch += QUEUESIZE
        client.reset()

client.reset()



'''
--------------------------------------------------------------------------------------------------------------    

INSPIRED FROM:
    Title: image_collection.py
    Availability: https://github.com/simondlevy/AirSimTensorFlow/blob/master/image_collection.py
    Authors: Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, Simon D. Levy, Will McMurtry, Jacob Rosen    
    
--------------------------------------------------------------------------------------------------------------
'''