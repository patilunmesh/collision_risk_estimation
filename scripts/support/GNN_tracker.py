'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
'''

#Code to track the detected objects using GNN and motion models
#Subscribes to detected objects then initializes tracklets 
#Uses GNN for second step and next step onwards combines GNN with the prediction step using Motion models
#jsk_recognition_msgs/BoundingBoxArray
# to trim arrays data = data[~np.all(data == 0, axis=1)]

#TODO
#Use of models to predict and check 
#KITTI bag has no clock so the header of stamped objects needs to be taken from somewhere
############################################################################################################
from numba import njit, jit, prange, vectorize
from numba import float64 as f64
from numba import int8 
import math, rospy, sys, tf
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from jsk_recognition_msgs.msg import BoundingBoxArray
from tf.transformations import euler_from_quaternion

############################################################################################################
init = True
rawList = BoundingBoxArray()
objectNew = np.zeros((20, 5), dtype=np.float64) # x, y , yaw , w, h
objectOld = np.zeros((20, 6), dtype=np.float64) # tracked objects with 6th field of id
count = 0
addr = 0
#############################################################################################################

#@njit(int8[:](f64[:, :], f64[:,:], int8, int8), nogil=True, fastmath=True, cache=True)
def GNN(old, new, f, oldf):
	global addr
	hypothesis = np.zeros((f), dtype=np.int8)
	distances = np.zeros((f)) + 10000
	#print('counts', oldf, f)
	for i in range(f):
		xn = new[i][0]
		yn = new[i][1]
		minD = 10000
		#print('iteration start', i)
		for j in range(oldf):
			x, y = old[j][0], old[j][1]
			#print(xn , yn, x, y, minD, i, j)
			if np.hypot(xn - x, yn - y) < minD:
				hypothesis[i] = old[j][5]
				minD = np.hypot(xn - x, yn - y)
				distances[i] = minD
	for k in range(f):
		for j in range(f):
			if (hypothesis[k] == hypothesis[j]) and k != j:
				#print(k, j, distances[k], distances[j], oldf, addr)
				if distances[j] >= distances[k]:
					hypothesis[j] = addr
					addr += 1
					#if addr == f + 1 - oldf: 
	return hypothesis

def orderingObjects(idList, dets):
	for i in range(len(idList)):
		dets.boxes[i].label = idList[i]
	return dets


#############################################################################################################

def det_sub(dets):
	global init, count, objectNew, objectOld, addr
	idList = []
	old_count = count
	count = len(dets.boxes)
	#print("oldcount", old_count, 'new count', count)
	if init:
		old_count = count
		init = False
		for n in range(count):
			dets.boxes[n].label = n
		objectPublisher(dets)
		addr = old_count
	else:
		for i in range(count):
			objectNew[i][0] = dets.boxes[i].pose.position.x
			objectNew[i][1] = dets.boxes[i].pose.position.y
			rot = []
			rot = [0, 0, dets.boxes[i].pose.orientation.z, dets.boxes[i].pose.orientation.w]
			(roll, pitch, yaw) = euler_from_quaternion(rot)
			objectNew[i][2] = yaw
			objectNew[i][3] = dets.boxes[i].dimensions.x
			objectNew[i][4] = dets.boxes[i].dimensions.y
		idList = GNN(objectOld, objectNew, count, old_count)
		obj_data = orderingObjects(idList, dets)
		objectPublisher(obj_data)
	#print('next iter')

def objectPublisher(obj_data):
	global objectOld
	rList = []
	rList = obj_data.boxes
	marker_array_ = MarkerArray()
	total = len(rList)
	#print(total)
	for i in range(total):
		marker_ = Marker()
		marker_.ns = "tracked_objects"
		marker_.id = rList[i].label
		marker_.header = rList[i].header
		marker_.type = 1 #cube
		marker_.action = marker_.ADD
		marker_.pose =  rList[i].pose
		marker_.lifetime = rospy.Duration.from_sec(1)
		marker_.scale.x = rList[i].dimensions.x
		marker_.scale.y = rList[i].dimensions.y
		marker_.scale.z = rList[i].dimensions.z
		marker_.color.a = 1.0
		marker_.color.b = 1/(rList[i].label + 1)
		marker_.color.g = 0.1*rList[i].label
		marker_.color.r = rList[i].label / (rList[i].label + 1000)
		marker_.text = "Car"
		marker_array_.markers.append(marker_)
	marker_pub.publish(marker_array_)
	objectOld *= 0
	for i in range(total):
		objectOld[i][0] = float(rList[i].pose.position.x)
		objectOld[i][1] = float(rList[i].pose.position.y)
		rot = []
		rot = [0, 0, rList[i].pose.orientation.z, rList[i].pose.orientation.w]
		(roll, pitch, yaw) = euler_from_quaternion(rot)
		objectOld[i][2] = yaw
		objectOld[i][3] = rList[i].dimensions.x
		objectOld[i][4] = rList[i].dimensions.y
		objectOld[i][5] = rList[i].label


############################################################################################################
if __name__ == '__main__':
    try:
    	rospy.init_node('tracker', anonymous=True)
    	rospy.Subscriber("/second_arr_bbox", BoundingBoxArray, det_sub)
    	marker_pub = rospy.Publisher("/filter_objects_data", MarkerArray, queue_size=10)
    	rospy.spin()
    except rospy.ROSInterruptException:
    	pass
