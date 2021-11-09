'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Calculates Displacement error (Avg.) in motion prediction using recorded CSV files of ground truth

'''
################################################################################### section 1 imports
from numba import njit
from numba import float64 as f64
import rospy, sys, tf, os
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
import csv

########################################################################### section 2 defaults and global vars
marker_array_ = MarkerArray()
Delt_Accum, AccumDev, AccumPclose, AccumCount = 0, 0 ,0, 0
angcons, lincons, angcons2 = 7.1, 6.1, 0
traj_data = []
xldata, yldata = [], []
x_data, y_data = [], []
time_data = []
Sequence = True
objectList =  np.zeros((12))  #[[0 for k in range(10)] for j in range(count)]
t = 2 #time horizon
pt = 0.99 #threshold
resolution = 1
vxmy, vymy, wmy = 0 , 0, 0
#car model constants (enclosing hull)
#c1_car = 6.1
MaxD = 10*t
psi = None						# Heading
lat0, lon0 = 0, 0
################################################################################# section 3 probability models

mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-4] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_FDEmean

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

def get_index(Future_time):
	global time_data
	mindiff = 100
	for i in range(len(time_data)):
		diff = abs(Future_time - time_data[i])
		if diff < mindiff:
			mindiff = diff
			index = i
	tester = abs(Future_time - time_data[index])
	if tester < 0.7:
		return index
	else:
		return -1

def auto_calibrator():
	global t
	global Delt_Accum, traj_data
	future_x, future_y = 0, 0
	Future_time = Delt_Accum + t
	ind = get_index(Future_time)
	if ind < len(traj_data) and (ind != -1):
		future_x = traj_data[ind][0]
		future_y = traj_data[ind][1]
	return future_x, future_y
					

def csv_reader():
	global traj_data, time_data
	csvpath = mypath[:-12] + 'csvdata/linear' + b +'.csv'
	with open (csvpath , 'r') as p:
		read_p = csv.reader(p)
		point_data = list(read_p)
		p.close()
	for i in range(len(point_data)):
		time_data.append(float(point_data[i][0]))
		traj_data.append((float(point_data[i][1]), float(point_data[i][2])))


############################################################################################# section 5 callbacks	
def parse_imu_data(msg):
	# Get yaw angle.
	global psi
	ori = msg.orientation
	quat = (ori.x, ori.y, ori.z, ori.w)
	roll, pitch, psi = euler_from_quaternion(quat)

def updatecons(): # grassfire search for caliberation. Use only once
	global angcons, angcons2, lincons
	if angcons < 20 and lincons < 20:
		lincons += 1
	if lincons > 19 and angcons < 20:
		angcons += 1
		lincons = 0.1
	if lincons > 19 and angcons > 19:
		print('############### experiment complete! #####################')

def updatetime(t, pt): # put the rosbag in loop and this will update time horizon and pthreshold. t= 1,2,3 pt = 0.8 to 0.99
	if t < 4 and pt <= 0.99:
		t += 1
		return t, pt
	if t == 4 and pt < 0.99:
		pt += 0.05
		if pt > 0.99: pt = 0.99
		t = 1
		return t, pt
	if t == 4 and pt == 0.99:
		print('############### experiment complete! #####################')
		return t, pt


def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

def parse_gps_fix(msg):
	# This function gets the latitude and longitude from the OxTS.
	global Sequence, lat0, lon0, Delt_Accum, pt
	global objectList, t, vxmy, vymy, wmy, psi, AccumDev, AccumPclose, AccumCount
	future_x, future_y = 0, 0
	if Sequence:
		Sequence = False
		lat0 = msg.latitude
		lon0 = msg.longitude
		objectList[0] = msg.header.stamp.to_sec()
		objectList[8] = vxmy 
		objectList[9] = vymy
	if not Sequence:
		lat = msg.latitude
		lon = msg.longitude
		X,Y = latlon_to_XY(lat, lon)
		DelT = msg.header.stamp.to_sec() - objectList[0]
		if 1 > DelT > 0:
			Delt_Accum += DelT
			objectList[0] =  msg.header.stamp.to_sec()
			objectList[1] = (vxmy - objectList[8])/ DelT 
			objectList[2] = (vymy - objectList[9])/ DelT 
			objectList[3] = X
			objectList[4] = Y
			objectList[5] = psi
			objectList[7] = wmy
			objectList[8] = vxmy 
			objectList[9] = vymy
			future_x, future_y = auto_calibrator()
			if future_x !=0 and future_y != 0:
				objectList[10] = future_x
				objectList[11] = future_y
				minD = prob_machine_FDEmean(objectList, t, pt)
				if minD < 6: 
					AccumCount += 1
					#AccumDev += highProbDist
					AccumPclose += minD
	if Delt_Accum > 35.1:
		#ADE = AccumDev / AccumCount
		AvgPclose = AccumPclose / AccumCount
		print('AvgPclose:', AvgPclose, 'threshold ', pt, 'time ', t, 'Delt acc', Delt_Accum)
		t, pt = updatetime(t, pt)
		#print('new tpt' , pt, t)
		Delt_Accum, AccumDev, AccumPclose, AccumCount = 0, 0 ,0, 0
		objectList *= 0
		Sequence = True



#################################################################################################### section 6 main body

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='validating R-space')
	parser.add_argument('-t', '--time-horizon', default='2',help='time horizon for plot')
	parser.add_argument('-b', '--bag-id', default='91',help='name of the bag')
	args = parser.parse_args()
	t = float(args.time_horizon)
	b = args.bag_id
	try:
		rospy.init_node('motion_models', anonymous=True)
		csv_reader()
		rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub)
		rospy.Subscriber('kitti/oxts/gps/fix', NavSatFix, parse_gps_fix, queue_size=1)
		rospy.Subscriber('/kitti/oxts/imu', Imu, parse_imu_data, queue_size=1)
		rospy.spin()		
	except rospy.ROSInterruptException:
		pass

