'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Prints the time elapsed for each iteration and the number of objects present in the scene.
'''
######################################################################### section1 imports
from numba import njit
from numba import float64 as f64
from numba import float32 as f32
from numba import int8
import math, rospy, tf
import numpy as np
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
import time
########################################################################### section 2 defaults and global vars
marker_array_ = MarkerArray()
Sequence = True;
marker_list = np.zeros((10, 5), dtype=np.float64) 
objectList =  np.zeros((10, 12), dtype=np.float64)
t = 2 #time horizon
resolution = 0.2
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
@njit(int8 (int8, f64[:,:], f32, f32, int8), nogil=True, fastmath=True, cache=True)
def prob_machine(f, objectList, vsx, vsy, t):
	global n, w, h, originY, originX, c1_car
	k = 100
	dynamic_count = 0
	#f= int(f)
	rspaceDat = np.zeros((w*h), dtype=np.float32)
	dx = vsx*n*t + 5
	for i in range(f):
		vx, vy = objectList[i][8], objectList[i][9]
		v = np.hypot(vx, vy)
		point_count = 1.0
		point_intr = 0.0
		if v > 1:
			x = (objectList[i][3] - 0.5 - originX) * n 
			y = (objectList[i][4] - originY)* n
			theta_given = objectList[i][5]
			sx = n*objectList[i][10]*0.5
			sy = n*objectList[i][11]*0.5 #h
			omega = 0#0.5*objectList[i][7]
			ax, ay, ttype = objectList[i][1], objectList[i][2], objectList[i][6]
			a = np.hypot(ax, ay)
			Vf = v + a*t #if v + a*t < 12 else 11
			Dx = x + vx*t + 0.5*ax*t*t
			Dy = y + vy*t + 0.5*ay*t*t
			D = (np.hypot(Dx, Dy))*0.2
			if D > MaxD: D = MaxD
			ymin = int(y - k) if (y - k) > originY else originY
			ymax = int(y + k) if (y + k) < w else w
			xmin = int(x - k) if (x - k) > originX else originX
			xmax = int(x + k) if (x + k) < h else h
			yg = np.arange(ymin, ymax) - y
			factor = v*t*((v-1)/(v+1)) + 0.5*a*t*t*((a-1)/(a+1))
			rspLocal = np.zeros((w*h), dtype=np.float32)
			if ttype < 3 :
				dynamic_count += 1
				for xg in range(xmin, xmax, 1):
					d1 = np.hypot(yg, xg-x) - D
					delth = np.arctan2(yg, xg-x) - theta_given - omega
					for j in range(len(d1)):
						y_ = int(yg[j] + y)
						deltha, d = delth[j], d1[j]
						if abs(theta_given) > 2.57: 
							if deltha > 6:
								deltha = deltha - 6.28
							elif deltha < -6:
								deltha = deltha + 6.28
						Pa = 1 - (deltha*deltha*Vf*10/ (t *t))
						cons = 1.2 if ttype > 1 else c1_car 
						Pl = 1 - (d*d / (cons*factor + 0.0001))
						if Pa < 0: Pa = 0
						if Pa > 1: Pa = 1
						if Pl < 0: Pl = 0
						p = Pa*Pl
						if p > 1: p = 1
						#rspLocal[h*y_ + xg] = p
						orient = 2*np.arctan((y_-y) / (xg-x)) - theta_given
						if p > 0.1:
							point_count += p
							cosi = np.cos(orient)
							sine = np.sin(orient)
							TRx = xg + (sx * cosi) - (sy * sine)
							TRy = y_ + (sx * sine) + (sy * cosi)
							TLx = xg - (sx * cosi) - (sy * sine)
							TLy = y_ - (sx * sine) + (sy * cosi)
							BLx = xg - (sx * cosi) + (sy * sine)
							BLy = y_ - (sx * sine) - (sy * cosi)
							BRx = xg + (sx * cosi) + (sy * sine)
							BRy = y_ + (sx * sine) - (sy * cosi)
							arx = np.array([ TRx, TLx, BLx, BRx])
							ary = np.array([ TRy, TLy, BLy, BRy])
							xma, xmi, yma, ymi = np.max(arx), np.min(arx), np.max(ary), np.min(ary)
							DT = (TRx - BRx) * (y_ - BRy) - (xg - BRx) * (TRy - BRy)
							DL = (TLx - BLx) * (y_ - BLy) - (xg - BLx) * (TLy - BLy)
							DR = (TLx - TRx) * (y_ - TRy) - (xg - TRx) * (TLy - TRy)
							DB = (BRx - BLx) * (y_ - BLy) - (xg - BLx) * (BRy - BLy)
							for xl in range(xmi, xma, 1):
								for yi in range(ymi, yma, 1):
									if (w > xl > 0) and (h > yi > 0):
										DTl = (TRx - BRx) * (yi - BRy) - (xl - BRx) * (TRy - BRy)
										if (DTl*DT) > 0:
											DLl = (TLx - BLx) * (yi - BLy) - (xl - BLx) * (TLy - BLy)
											if (DLl*DL > 0):
												DRl = (TLx - TRx) * (yi - TRy) - (xl - TRx) * (TLy - TRy)
												if (DRl*DR) > 0:
													DBl = (BRx - BLx) * (yi - BLy) - (xl - BLx) * (BRy - BLy)
													if (DBl*DB) > 0:
														rspLocal[h*yi + xl] += p
														if (0 < xg < dx):
															yrange = 400 + (xg*xg*vsy / (26*vsx + 0.01))
															if (yrange - 10 < y_ < yrange+10):
																if rspLocal[h*yi + xl]  > point_intr: point_intr = rspLocal[h*yi + xl]
											

	return dynamic_count
#################################################################################### section 4 helper functions

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
				ttype = 0.0
				if marker_data.markers[i].text == "Car":
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
		begt = time.time()
		rsp = prob_machine(car_count, objectList, vxmy, vymy, t)
		end = time.time()
		print((end-begt), car_count)

def vel_sub(vel):
	global vxmy, vymy, wmy
	vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
	wmy = vel.twist.angular.z

#################################################################################################### section 6 main body

if __name__ == '__main__':
	try:
		rospy.init_node('motion_models', anonymous=True)
		rospy.Subscriber("/kitti/tracklet", MarkerArray, callback_sub)
		rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub)
		rospy.spin()
	except rospy.ROSInterruptException:
		pass