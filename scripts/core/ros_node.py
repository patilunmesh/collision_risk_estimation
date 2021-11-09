'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Estimates motion of kKITTI tracklet and predicts future occupancy using stochastic models.
'''

################################################################################### section 1 imports
import rospy
import tf
import numpy as np
from visualization_msgs.msg import MarkerArray
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
import argparse
from probability_machine import prob_machine_gridgen

########################################################################### section 2 defaults and global vars
marker_array_ = MarkerArray()
Sequence = True;
marker_list = np.zeros((20, 5)) #[[0 for k in range(6)] for j in range(count)]
objectList =  np.zeros((20, 12))  #[[0 for k in range(10)] for j in range(count)]
t = 2 #time horizon
resolution = 0.1
n = int(1/resolution)
w = 80*n + 1
h = 80*n + 1
originX = 0
originY = -40
vxmy, vymy, wmy = 0 , 0, 0

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
	gridmap.info.pose.position.x = -originY
	gridmap.info.pose.position.y = originX
	gridmap.info.header.frame_id = "kitti/base_link"
	gridmap.info.resolution = 0.1
	grid_pub.publish(gridmap)

############################################################################################# section 5 callbacks	

def callback_sub(marker_data):
	global Sequence, marker_list, objectList, t
	global vxmy, vymy, wmy
	count = len(marker_data.markers)
	car_count = 0
	if Sequence:
		Sequence = False
		for i in range(count):
				marker_list[i][0] = marker_data.markers[i].header.stamp.to_sec()
				marker_list[i][1] = marker_data.markers[i].id
				marker_list[i][2] = marker_data.markers[i].pose.position.x 
				marker_list[i][3] = marker_data.markers[i].pose.position.y 
				marker_list[i][4] = marker_data.markers[i].pose.orientation.z		
	else:
		Sequence = True
		for i in range(count):
			if (marker_data.markers[i].id == marker_list[i][1]):
				car_count += 1
				ttype = 3.0
				if marker_data.markers[i].text == "Car": ttype = 1.0
				if marker_data.markers[i].text == "Cyclist": ttype = 2.0
				if marker_data.markers[i].text == "Pedestrian": ttype = 4.0
				DelT = marker_data.markers[i].header.stamp.to_sec() - marker_list[i][0]
				x_ = marker_data.markers[i].pose.position.x
				y_ = marker_data.markers[i].pose.position.y
				vx = ( x_ - marker_list[i][2])/ (DelT)
				vy = ( y_ - marker_list[i][3])/ (DelT)
				#relative velocity
				vx = (vxmy - abs(vx))*abs(vx)/vx
				vy = (vymy - abs(vy))*abs(vy)/vy 
				rot = []
				rot = [0, 0, marker_data.markers[i].pose.orientation.z, marker_data.markers[i].pose.orientation.w]
				(roll, pitch, yaw) = euler_from_quaternion(rot)
				omega = (yaw - objectList[i][5])/DelT
				objectList[i][0] = marker_data.markers[i].id
				objectList[i][1] = (vx - objectList[i][8])/ DelT #ax
				objectList[i][2] = (vy - objectList[i][9])/ DelT #ay
				objectList[i][3] = x_
				objectList[i][4] = y_
				objectList[i][5] = yaw
				objectList[i][6] = ttype
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
	t = float(args.time_horizon)
	try:
		rospy.init_node('motion_models', anonymous=True)
		rospy.Subscriber("/kitti/tracklet", MarkerArray, callback_sub)
		rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub)
		grid_pub = rospy.Publisher("/rspaceGrid2", GridMap, queue_size=1 )
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

