'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Shows predicted future occupancy for a scenario in carla simulator.
First play the bagfile, then run the carla_objects code and then orsp_carla.py. Open rviz config to visualize.
'''
################################################################################### section 1 imports
import math, rospy, sys, tf, os
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
from tf import TransformListener
import argparse
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

########################################################################### section 2 defaults and global vars
marker_array_ = MarkerArray()
selfid = 606
Sequence = True;
objectList =  np.zeros((10,12), dtype=np.float64)
marker_list = np.zeros((10, 5), dtype=np.float64)
t = 2 #time horizon
resolution = 0.1
n = int(1/resolution)
w = 80*n + 1
h = 80*n + 1
originX = 0
originY = 0
vxmy, vymy, wmy = 1.0 , 1.0, 0.0

################################################################################# section 3 probability models
mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-5] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_gridgen

#################################################################################### section 4 helper functions

def gridMap_generator(rspaceData):
	global resolution, n, w, h, originX, originY
	rspaceData = np.asarray(rspaceData, dtype=np.float64).reshape(h,w)
  	rspaceData = np.rot90(rspaceData, 2).flatten().tolist()
	gridmap = GridMap()
	multi_array = Float32MultiArray()
	multi_array.layout.dim.append(MultiArrayDimension())
	multi_array.layout.dim.append(MultiArrayDimension())
	multi_array.layout.dim[0].label = "column_index"
	multi_array.layout.dim[0].size = w
	multi_array.layout.dim[0].stride = h*w
	multi_array.layout.dim[1].label = "row_index"
	multi_array.layout.dim[1].size = h
	multi_array.layout.dim[1].stride = w
	multi_array.data = rspaceData
	gridmap.layers.append("elevation")
	gridmap.data.append(multi_array)
	gridmap.info.length_x = 80
	gridmap.info.length_y = 80
	gridmap.info.pose.position.x = originY + 40
	gridmap.info.pose.position.y = originX + 40
	gridmap.info.header.frame_id = "/hero"
	gridmap.info.resolution = 0.1
	grid_pub.publish(gridmap)


############################################################################################# section 5 callbacks	

once = True
def callback_sub(marker_data):
	global Sequence, marker_list, objectList, t
	global vxmy, vymy, wmy, x_data, y_data, once, selfid, bagtimez
	count = len(marker_data.markers)
	car_count = 0.0
	if Sequence:
		Sequence = False
		for i in range(count):
			#print(marker_data.markers[i].id)
			if marker_data.markers[i].id != selfid:
				marker_list[i][0] = marker_data.markers[i].header.stamp.to_sec()
				marker_list[i][1] = marker_data.markers[i].id
				marker_list[i][2] = marker_data.markers[i].pose.position.x 
				marker_list[i][3] = marker_data.markers[i].pose.position.y 
				marker_list[i][4] = marker_data.markers[i].pose.orientation.z		
	else:
		Sequence = True
		for i in range(count):
			if (marker_data.markers[i].id == marker_list[i][1] and marker_data.markers[i].id != selfid):
				car_count += 1
				DelT = marker_data.markers[i].header.stamp.to_sec() - marker_list[i][0]
				x_ = marker_data.markers[i].pose.position.x
				y_ = marker_data.markers[i].pose.position.y
				vx = ( x_ - marker_list[i][2])/ (DelT)
				vy = ( y_ - marker_list[i][3])/ (DelT)
				#relative velocity
				vx += vxmy
				vy += vymy
				rot = []
				rot = [0, 0, marker_data.markers[i].pose.orientation.z, marker_data.markers[i].pose.orientation.w]
				(roll, pitch, yaw) = euler_from_quaternion(rot)
				omega = (yaw - objectList[i][5])/DelT
				#relative omega
				omega = (wmy + omega)
				objectList[i][0] = marker_data.markers[i].id
				objectList[i][3] = x_
				objectList[i][4] = y_
				objectList[i][5] = yaw
				objectList[i][6] = 4
				objectList[i][7] = omega
				objectList[i][8] = vx 
				objectList[i][9] = vy
				objectList[i][10] = marker_data.markers[i].scale.x #w
				objectList[i][11] = marker_data.markers[i].scale.y #h
		rsp = prob_machine_gridgen(car_count,originX, originY, objectList, vxmy, vymy, t)
		gridMap_generator(rsp)

def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

#################################################################################################### section 6 main body

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='validating R-space')
	parser.add_argument('-t', '--time-horizon', default='2',help='time horizon for plot')
	args = parser.parse_args()
	t = float(args.time_horizon) #time horizon
	try:
		rospy.init_node('motion_models', anonymous=True)
		rospy.Subscriber("/filter_objects_data", MarkerArray, callback_sub)
		rospy.Subscriber("/self_vel", TwistStamped, vel_sub)
		grid_pub = rospy.Publisher("/rspaceGrid2", GridMap, queue_size=1 )
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

