'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: collision risk plot for a scenario in carla. Specifically for the provided bagfile. (car_cross) 
For other bagfiles, check the objectid, bagtime, duration, trackid etc. Check argument parser.
'''

################################################################################### section 1 imports
import rospy, sys, tf, os
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
from tf import TransformListener
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
########################################################################### section 2 defaults and global vars
fig, ax = plt.subplots()
ur, = plt.plot([], [], 'k-', linewidth=2, label='risk')
#up, = plt.plot([], [])
x_data, y_data = [] , []
trackid = 0
T_init = 0.0
bagtimez = 38.1885#14.646352
duration_bag = 10
marker_array_ = MarkerArray()
Sequence = True;
objectList =  np.zeros((12), dtype=np.float64)
t = 2 #time horizon
resolution = 0.1
n = int(1/resolution)
w = 80*n + 1
h = 80*n + 1
originX = 0
originY = 0 #-40 ####set this to zero to deal with recoreded bags
vxmy, vymy, wmy = 1.0 , 1.0, 0.0
once = True
################################################################################# section 3 probability models

mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-5] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_cplotter

############################################################################################# section 4 callbacks	
def callback_sub(marker_data):
	global Sequence, objectList, t
	global vxmy, vymy, wmy, timekeep, x_data, y_data, once, trackid, bagtimez, T_init
	count = len(marker_data.markers)
	for i in range(count):
		if marker_data.markers[i].id ==trackid:
			DelT =  marker_data.markers[i].header.stamp.to_sec() - T_init
			T_init = marker_data.markers[i].header.stamp.to_sec()
			rot = []
			rot = [0, 0, marker_data.markers[i].pose.orientation.z, marker_data.markers[i].pose.orientation.w]
			(roll, pitch, yaw) = euler_from_quaternion(rot)
			if Sequence:
				Sequence = False
				objectList[3] = marker_data.markers[i].pose.position.x
				objectList[4] = marker_data.markers[i].pose.position.y
				objectList[5] = yaw
			if not Sequence:
				x_ = marker_data.markers[i].pose.position.x
				y_ = marker_data.markers[i].pose.position.y
				#print(x_, y_)
				vx = ( x_ - objectList[3])/ (DelT)
				vy = ( y_ - objectList[4])/ (DelT)
				vx += vxmy
				vy += vymy
				omega = (yaw - objectList[5])/DelT
				omega = (wmy + omega)
				objectList[1] = (vx - objectList[8])/ DelT #ax
				objectList[2] = (vy - objectList[9])/ DelT #ay
				objectList[3] = x_
				objectList[4] = y_
				objectList[5] = yaw
				objectList[7] = omega
				objectList[8] = vx 
				objectList[9] = vy
				objectList[10] = marker_data.markers[i].scale.x #w
				objectList[11] = marker_data.markers[i].scale.y #h
				ret = prob_machine_cplotter(1, originX, originY, objectList, vxmy, vymy, t)
				if ret > 0: once = False
				if once:
					y_data.append(ret)
					x_data.append(T_init -  bagtimez)
				if not once:
					if ret > 0.3:
						y_data.append(ret)
						x_data.append(T_init -  bagtimez)
					if T_init - bagtimez > 8.2:
						y_data.append(ret)
						x_data.append(T_init -  bagtimez)


def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

#################################################################################################### section 6 main body
def plot_init():
	global duration_bag
	ax.set_xlim(0, duration_bag)
	ax.set_ylim(0, 1.2)
	plt.xlabel("Time")
	plt.ylabel("Risk")
	title = 'RS Collision risk ' + str(t) + 's'
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
	parser.add_argument('-d', '--duration', default='10',help='duration of bag')
	parser.add_argument('-i', '--init', default='38.1885',help='first time stamp')
	parser.add_argument('-tid', '--tid', default='0',help='ID of object to track')
	args = parser.parse_args()
	t = float(args.time_horizon) #time horizon
	duration_bag = float(args.duration)
	bagtimez = float(args.init)
	trackid = int(args.tid)
	try:
		print('if it is recorded bag check origin, time, duration, trackid')
		rospy.init_node('risk_plotter', anonymous=True)
		rospy.Subscriber("/filter_objects_data", MarkerArray, callback_sub)
		rospy.Subscriber("/self_vel", TwistStamped, vel_sub)
		ani = FuncAnimation(fig, update_plot, init_func=plot_init)
		plt.show(block=True)
		plt.pause(0.01)
	except rospy.ROSInterruptException:
		pass