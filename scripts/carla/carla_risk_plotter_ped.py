'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: collision risk plot for a scenario in carla. Specifically for the provided bagfile. (pedstr_crash)
For other bagfiles, check the objectid, bagtime, duration, trackid etc. Check argument parser.
'''

################################################################################### section 1 imports
import math, rospy, sys, tf
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
from tf import TransformListener
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse, os
########################################################################### section 2 defaults and global vars
fig, ax = plt.subplots()
ur, = plt.plot([], [], linewidth=2)
#up, = plt.plot([], [])
x_data, y_data, ytdata = [] , [], []
selfid = 606 #2933
bagtimez = 14.796351946
duration_bag = 12
marker_array_ = MarkerArray()
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
#car model constants (enclosing hull)
c1_car = 2.4

################################################################################# section 3 probability models

mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-5] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_pedc
#################################################################################### section 4 helper functions

def grid_generator(rspaceData):
	global resolution, n, w, h, originX, originY
	grid = VelocityGrid()
	grid.header.seq = 0 
	grid.header.stamp = rospy.Time.now()
	grid.header.frame_id = "/hero"
	grid.info.map_load_time = rospy.Duration.from_sec(0)
	grid.info.resolution = resolution
	grid.info.width = w
	grid.info.height = h
	grid.info.origin.position.x = originX
	grid.info.origin.position.y = originY
	grid.info.origin.position.z = 0.0
	grid.info.origin.orientation.x = grid.info.origin.orientation.y = grid.info.origin.orientation.z = 0.0
	grid.info.origin.orientation.w = 1.0
	grid.nb_channels = 1
	grid.data = rspaceData.tolist()
	grid_pub.publish(grid)


############################################################################################# section 5 callbacks	
def callback_sub(marker_data):
	global Sequence, marker_list, objectList, t
	global vxmy, vymy, wmy, x_data, y_data, once, selfid, bagtimez
	count = len(marker_data.markers)
	car_count = 0
	if Sequence:
		Sequence = False
		for i in range(count):
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
				objectList[i][6] = 1
				objectList[i][7] = omega
				objectList[i][8] = vx 
				objectList[i][9] = vy
				objectList[i][10] = marker_data.markers[i].scale.x #w
				objectList[i][11] = marker_data.markers[i].scale.y #h
		ret = prob_machine_pedc(car_count, originX, originY, objectList, vxmy, vymy, t)
		#print(rospy.get_time(), bagtimez, ret)
		y_data.append(ret)
		x_data.append(rospy.get_time() - bagtimez)


def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

#################################################################################################### section 6 main body
def plot_init():
	global duration_bag
	ax.set_xlim(0, duration_bag)
	ax.set_ylim(0, 1)
	plt.xlabel("Time")
	plt.ylabel("Risk")
	title = 'Collision risk ' + str(t) + 's'
	plt.title(title)
	plt.yticks(np.arange(0, 1.2, 0.1))
	return ur

def update_plot(frame):
    ur.set_data(x_data, y_data)
    #up.set_data(x_data, ytdata)
    return ur

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='validating R-space')
	parser.add_argument('-t', '--time-horizon', default='2',help='time horizon for plot')
	args = parser.parse_args()
	t = float(args.time_horizon) #time horizon
	try:
		rospy.init_node('risk_plotter', anonymous=True)
		rospy.Subscriber("/filter_objects_data", MarkerArray, callback_sub)
		rospy.Subscriber("/self_vel", TwistStamped, vel_sub)
		ani = FuncAnimation(fig, update_plot, init_func=plot_init)
		plt.show(block=True)
		plt.pause(0.01)
	except rospy.ROSInterruptException:
		pass