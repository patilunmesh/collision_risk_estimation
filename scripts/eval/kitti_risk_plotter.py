'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Estimates motion of kKITTI tracklet and predicts future occupancy using stochastic models. 
The predicted occupancy is then used to estimate the collision risk, which is plotted with time.
'''

################################################################################### section 1 imports
import math, rospy, sys, tf, os
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
########################################################################### section 2 defaults and global vars
fig, ax = plt.subplots()
ur, = plt.plot([], [], 'k-', linewidth=2, label='2s risk')
#up, = plt.plot([], [])
x_data, y_data = [] , []
bagtimez = 37.688577#14.646352
duration_bag = 10
once = True
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
#car model constants (enclosing hull)
c1_car = 2.4
MaxD = 10*n*t #m
aSetCar = 1.3 #m/s2 setCar indicates the bounds above which car can be consideed as  dynamic
vSetCar = 1 #m/s
################################################################################# section 3 probability models
mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-4] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_riskgen

#################################################################################### section 4 helper functions
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


############################################################################################# section 5 callbacks	

def callback_sub(marker_data):
	global Sequence, marker_list, objectList, t, x_data, y_data,bagtimez
	global vxmy, vymy, wmy
	count = len(marker_data.markers)
	maxret = []
	car_count = 0
	if Sequence:
		Sequence = False
		for i in range(count):
			marker_list[i][0] = marker_data.markers[i].header.stamp.to_sec()
			#print(marker_list[i][0])
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
				if marker_data.markers[i].text == "Car" or marker_data.markers[i].text == "Truck":
					ttype = 1.0
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
				#relative omega
				omega = (wmy + omega)
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
		ret = prob_machine_riskgen(car_count, originX, originY, objectList, vxmy, vymy, t)
		maxret.append(ret)
	if len(maxret) > 0:
		y_data.append(max(maxret))
		x_data.append(marker_data.markers[i].header.stamp.to_sec() -  bagtimez)
		maxret *= 0


def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

#################################################################################################### section 6 main body

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='validating R-space')
	parser.add_argument('-t', '--time-horizon', default='2',help='time horizon for plot')
	parser.add_argument('-d', '--duration', default='35',help='duration of bag')
	parser.add_argument('-i', '--init', default='1317050024.342104',help='first time stamp')
	args = parser.parse_args()
	t = float(args.time_horizon)
	duration_bag = float(args.duration)
	bagtimez = float(args.init)
	try:
		rospy.init_node('motion_models', anonymous=True)
		rospy.Subscriber("/kitti/tracklet", MarkerArray, callback_sub)
		rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub)
		ani = FuncAnimation(fig, update_plot, init_func=plot_init)
		plt.show(block=True)
		plt.pause(0.01)
	except rospy.ROSInterruptException:
		pass
