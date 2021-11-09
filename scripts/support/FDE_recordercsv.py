'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Records trajectory of Ego vehicle in a CSV file along with the time stamp. Used to evaluate spatial deviation.
'''
################################################################################### section 1 imports
import numpy as np
import os, rospy
import math as m
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
x_data, y_data = [], []
time_data = []
Sequence = True
objectList =  np.zeros((12))  #[[0 for k in range(10)] for j in range(count)]
psi = None						# Heading
lat0, lon0 = 0, 0
Delt_Accum = 0
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

############################################################################################# section 5 callbacks	


def parse_gps_fix(msg):
	# This function gets the latitude and longitude from the OxTS.
	global Sequence, lat0, lon0, x_data, y_data, time_data
	global objectList, Delt_Accum
	if Sequence:
		Sequence = False
		lat0 = msg.latitude
		lon0 = msg.longitude
		objectList[0] = msg.header.stamp.to_sec()
	if not Sequence:
		lat = msg.latitude
		lon = msg.longitude
		X,Y = latlon_to_XY(lat, lon)
		DelT = msg.header.stamp.to_sec() - objectList[0]
		if DelT > 0:
			Delt_Accum += DelT
			x_data.append(X)
			y_data.append(Y)
			time_data.append(Delt_Accum)
			objectList[0] =  msg.header.stamp.to_sec()
			

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
		rospy.Subscriber('kitti/oxts/gps/fix', NavSatFix, parse_gps_fix, queue_size=1)
		rospy.spin()
		mypath = os.path.dirname(os.path.abspath(__file__))
		csvpath = mypath[:-15] + 'csvdata/linear' + b +'.csv'
		with open (csvpath, 'a') as t1:
			t1w = csv.writer(t1)
			for z in range(len(x_data)):
				t1w.writerow([time_data[z], x_data[z], y_data[z]])
		t1.close()
	except rospy.ROSInterruptException:
		pass

