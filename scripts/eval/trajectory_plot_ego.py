'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Estimates motion of Ego vehicle and predicts future motion. 
The ground truth trajectory is plotted along with linear prediction and trajectory from prediction models developed in the project.
'''

################################################################################### section 1 imports
from numba import njit, jit, prange, vectorize
from numba import float64 as f64
import rospy, sys, tf
import math as m
import numpy as np
from sensor_msgs.msg import Imu
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
import argparse
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistWithCovarianceStamped
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

########################################################################### section 2 defaults and global vars
marker_array_ = MarkerArray()
x_data, y_data = [], []
xrdata, yrdata = [], []
xldata, yldata = [], []
occ_dev, lin_dev = [], []
tdata = []
fig, ax = plt.subplots()
us, = plt.plot([], [], 'y+', linewidth=2, label='Stochastic models')
up, = plt.plot([], [], 'b-', linewidth=2, label='Linear projection')
ur, = plt.plot([], [], 'k-', linewidth=2, label='Ground truth')
Sequence = True
objectList =  np.zeros((12))  #[[0 for k in range(10)] for j in range(count)]
t = 2 #time horizon
vxmy, vymy, wmy = 0 , 0, 0
#car model constants (enclosing hull)
c1_car = 2.4
MaxD = 10*t
aSetCar = 1.3 #m/s2 setCar indicates the bounds above which car can be consideed as  dynamic
vSetCar = 1 #m/s
psi = None						# Heading
lat0, lon0 = 0, 0
################################################################################# section 3 probability models

@njit(f64[:](f64[:], f64), nogil=True, fastmath=True, cache=True)
def prob_machine(objectList, t):
	global c1_car
	k = 70
	rspaceDat = np.zeros((40000), dtype=np.float64)
	x, y, vx, vy, theta_given = objectList[3], objectList[4], objectList[8], objectList[9], objectList[5]
	v = np.hypot(vx, vy)
	point_count = 1
	if v > 1:
		omega = objectList[7]
		ax, ay = objectList[1], objectList[2]
		a = np.hypot(ax, ay)
		Vf = v + a*t 
		Dx = vx*t + 0.5*ax*t*t
		Dy = vy*t + 0.5*ay*t*t
		D = (np.hypot(Dx, Dy))
		ymin = int(y - k) #if (y - k) > originY else originY
		ymax = int(y + k) #if (y + k) < w else w
		xmin = int(x - k) #if (x - k) > originX else originX
		xmax = int(x + k) #if (x + k) < h else h
		yg = np.arange(ymin, ymax) - y
		factor = v*t*((v-1)/(v+1)) + 0.5*a*t*t*((a-1)/(a+1))
		deviation = 0.0
		for xg in range(xmin, xmax, 1):
			delth = np.arctan2(yg, xg-x) - theta_given - omega
			for j in range(len(delth)):
				y_ = yg[j] + y
				deltha = delth[j]
				d = np.hypot(y_ - y, xg -x) - D
				Pa = 1 - (deltha*deltha*Vf*7.1/ (abs(omega)*t *t))
				Pl = 1 - (d*d / (5.1*factor + 0.0001))
				if Pa < 0: Pa = 0
				if Pa > 1: Pa = 1
				if Pl < 0: Pl = 0
				p = Pa*Pl
				if p > 1: p = 1
				if p > 0.5: #plotter
					rspaceDat[point_count] = xg
					rspaceDat[point_count + 1] = y_
					point_count += 2
	rspaceDat[0] = point_count
	return rspaceDat

#################################################################################### section 4 helper functions
def latlon_to_XY(lat1, lon1):
	global lat0, lon0
	R_earth = 6371000 # meters
	delta_lat = m.radians(lat1 - lat0)
	delta_lon = m.radians(lon1 - lon0)
	lat_avg = 0.5 * ( m.radians(lat1) + m.radians(lat0) )
	X = R_earth * delta_lon * m.cos(lat_avg)
	Y = R_earth * delta_lat
	return X,Y

def xygen(rsp):
    global xrdata, yrdata
    total = rsp[0]
    i = 1
    while i < total:
        xrdata.append(rsp[i])
        yrdata.append(rsp[i+1])
        i = i+2  

def plot_init():
	global x_data, y_data
	ax.set_xlim(-200, 50)
	ax.set_ylim(-10, 200)
	plt.xlabel("x")
	plt.ylabel("y")
	title = 'Motion model validation ' + str(t) + 's'
	plt.title(title)
	plt.legend(loc='upper left', fontsize='xx-large')
	return us, up, ur

def update_plot(frame):
    us.set_data(xrdata, yrdata)
    ur.set_data(x_data, y_data)
    up.set_data(xldata, yldata)
    return us, up, ur

############################################################################################# section 5 callbacks	
def parse_imu_data(msg):
	# Get yaw angle.
	global psi
	ori = msg.orientation
	quat = (ori.x, ori.y, ori.z, ori.w)
	roll, pitch, psi = euler_from_quaternion(quat)

def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

def parse_gps_fix(msg):
	# This function gets the latitude and longitude from the OxTS.
	global Sequence, lat0, lon0, x_data, y_data, tdata
	global objectList, t, vxmy, vymy, wmy, psi, occ_dev, lin_dev
	future_x, future_y = 0, 0
	if Sequence:
		Sequence = False
		lat0 = msg.latitude
		lon0 = msg.longitude
	lat = msg.latitude
	lon = msg.longitude
	X,Y = latlon_to_XY(lat, lon)
	DelT = msg.header.stamp.to_sec() - objectList[0]
	tdata.append(msg.header.stamp.to_sec())
	x_data.append(X)
	y_data.append(Y)
	xlp = X + (X - objectList[3])*t/DelT
	ylp = Y + (Y - objectList[4])*t/DelT
	xldata.append(xlp)
	yldata.append(ylp)
	ind = int(msg.header.stamp.to_sec() - tdata[0] + t)*10 #tinit and tdata 0 are same so dev is wrong
	objectList[0] =  msg.header.stamp.to_sec()
	objectList[1] = (vxmy - objectList[8])/ DelT 
	objectList[2] = (vymy - objectList[9])/ DelT 
	objectList[3] = X
	objectList[4] = Y
	objectList[5] = psi
	objectList[7] = wmy
	objectList[8] = vxmy 
	objectList[9] = vymy
	rsp = prob_machine(objectList, t)
	xygen(rsp)
	
#################################################################################################### section 6 main body

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='validating R-space')
	parser.add_argument('-t', '--time-horizon', default='2',help='time horizon for plot')
	args = parser.parse_args()
	t = float(args.time_horizon)
	try:
		rospy.init_node('motion_models', anonymous=True)
		rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub)
		rospy.Subscriber('kitti/oxts/gps/fix', NavSatFix, parse_gps_fix, queue_size=1)
		rospy.Subscriber('/kitti/oxts/imu', Imu, parse_imu_data, queue_size=1)
		ani = FuncAnimation(fig, update_plot, init_func=plot_init)
		plt.show(block=True)
		plt.pause(0.01)
	except rospy.ROSInterruptException:
		pass

