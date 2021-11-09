'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Estimates motion of KITTI tracklet and predicts future occupancy using stochastic models. 
Predicted occupancy of a car, a cycle and a pedestrian are plotted along with ground truth and linear projection for comparison.
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
ur, = plt.plot([], [], 'g+', linewidth=1, label='Motion models')
us, = plt.plot([], [], 'bo', linewidth = 1, label='Linear projection')
ln, = plt.plot([], [], 'ko',  linewidth=0.01, label='Ground truth')
x_data, y_data = [] , []
xrdata, yrdata = [], []
xldata, yldata = [], []
marker_array_ = MarkerArray()
Sequence = True;
objectList =  np.zeros((7,12), dtype=np.float32)  #[[0 for k in range(10)] for j in range(count)]
resolution = 0.1
t = 2
n = int(1/resolution)
w = 80*n + 1
h = 80*n + 1
originX = 0
originY = -40
vxmy, vymy, wmy = 0 , 0, 0
#car model constants (enclosing hull)
c1_car = 2.4
MaxD = 8*n*t #m
aSetCar = 1.3 #m/s2 setCar indicates the bounds above which car can be consideed as  dynamic
vSetCar = 1 #m/s
T_init = 0.0
################################################################################# section 3 probability models

mypath = os.path.dirname(os.path.abspath(__file__))
corepath = mypath[:-4] + 'core'
sys.path.insert(0, corepath)
from probability_machine import prob_machine_tplotter
        
#################################################################################### section 4 helper functions

def xygen(rsp):
    global xrdata, yrdata
    total = rsp[0]
    i = 1
    while i < total:
        xrdata.append(rsp[i])
        yrdata.append(rsp[i+1])
        i = i+2
        

############################################################################################# section 5 callbacks   

def callback_sub(marker_data):
    global Sequence, objectList, resolution, n, originX, originY, T_init
    global vxmy, vymy, wmy, x_data, y_data, yldata, xldata,t
    count = len(marker_data.markers)
    idlist = np.zeros((count), dtype=np.int8)
    for j in range(count):
        if j == 0:
            DelT =  marker_data.markers[j].header.stamp.to_sec() - T_init
            T_init = marker_data.markers[j].header.stamp.to_sec()
        idlist[j] = marker_data.markers[j].id
    if Sequence:
        Sequence = False
        for i in range(count):
            j = idlist[i]
            if marker_data.markers[i].id == j:
                objectList[j][3] = (marker_data.markers[i].pose.position.x - 0.5 - originX) * n 
                objectList[j][4] = (marker_data.markers[i].pose.position.y  - originY)* n
                rot = []
                rot = [0, 0, marker_data.markers[i].pose.orientation.z, marker_data.markers[i].pose.orientation.w]
                (roll, pitch, yaw) = euler_from_quaternion(rot)
                objectList[j][5] = yaw     
    else:
        for i in range(count):
            j = idlist[i]
            if (marker_data.markers[i].id == j): 
                x_ = (marker_data.markers[i].pose.position.x - 0.5 - originX) * n
                y_ = (marker_data.markers[i].pose.position.y - originY)* n
                x_data.append(x_)
                y_data.append(y_)
                xldata.append(x_ + (x_ - objectList[j][3])*t/DelT)
                yldata.append(y_ + (y_ - objectList[j][4])*t/DelT)
                ttype = 0.0
                vx = ( x_ - objectList[j][3])/ DelT
                vy = ( y_ - objectList[j][4])/ DelT
                rot = []
                rot = [0, 0, marker_data.markers[i].pose.orientation.z, marker_data.markers[i].pose.orientation.w]
                (roll, pitch, yaw) = euler_from_quaternion(rot)
                omega = (yaw - objectList[j][5])/DelT
                #omega = (wmy + omega)
                if marker_data.markers[i].text == "Cyclist": ttype = 4.0
                if marker_data.markers[i].text == "Pedestrian": ttype = 3.0
                objectList[j][1] = 0.1*(vx - objectList[j][8])/ DelT #ax
                objectList[j][2] = 0.1*(vy - objectList[j][9])/ DelT #ay
                objectList[j][3] = x_
                objectList[j][4] = y_
                objectList[j][5] = yaw
                objectList[j][6] = ttype
                objectList[j][7] = omega
                objectList[j][8] = vx
                objectList[j][9] = vy
                objectList[j][10] = marker_data.markers[i].scale.x #w
                objectList[j][11] = marker_data.markers[i].scale.y #h
        rsp = prob_machine_tplotter(objectList, idlist, vxmy, vymy, t, originX, originY)
        xygen(rsp)

def vel_sub(vel):
    global vxmy, vymy, wmy
    vxmy, vymy = vel.twist.linear.x, vel.twist.linear.y
    wmy = vel.twist.angular.z

# #################################################################################################### section 6 main body
def plot_init():
    #initiates a plot for x and y with grid dimensions
    global h, w, t
    ax.set_xlim(0, h)
    ax.set_ylim(0, w)
    plt.xlabel("x")
    plt.ylabel("y")
    title = 'Validation of Motion Models '+ str(t) + 's'
    plt.title(title)
    plt.legend(loc='upper right', fontsize='xx-large')
    return us, ln, ur

def update_plot(frame):
    ur.set_data(xrdata, yrdata) # reachability data
    us.set_data(xldata, yldata) # Linear projected data
    ln.set_data(x_data, y_data) # Ground truth
    return us, ln, ur

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validating R-space')
    parser.add_argument('-t', '--time-horizon', default='2',
                        help='time horizon for plot')
    args = parser.parse_args()
    t = float(args.time_horizon) #time horizon
    try:
        rospy.init_node('motion_models', anonymous=True)
        rospy.Subscriber("/kitti/tracklet", MarkerArray, callback_sub) #kitti objects topic from bag
        rospy.Subscriber("/kitti/oxts/gps/vel", TwistStamped, vel_sub) #kitti gps velocity of ego veh.
        ani = FuncAnimation(fig, update_plot, init_func=plot_init)
        plt.show(block=True)
        plt.pause(0.001)
    except rospy.ROSInterruptException:
        pass