'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
@description: check out my paper: https://hal.inria.fr/hal-03416222 
About code: Converts carla objects into the format required for the code, (ObjectArray to MarkerArray conversion)
This code works for the bag file uploaded on the google drive(car_cross). For other bag files, check origin, object id, frames etc.
'''
########################################### section 1 imports
import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from derived_object_msgs.msg import ObjectArray
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
from tf import TransformListener

############################################ section 2 Callbacks and processors
self_pose = Marker()
once = False
yawold = 0.0
def vel_sub(vel):
	global self_pose, once, yawold
	if once:
		pose = vel.pose
		DelT = vel.header.stamp.stamp.to_sec() - self_pose.header.stamp.stamp.to_sec()
		rot = []
		rot = [0, 0, pose.orientation.z, pose.orientation.w]
		(roll, pitch, yaw) = euler_from_quaternion(rot)
		twi = TwistStamped()
		twi.header = vel.header
		twi.twist.linear.x = abs(pose.position.x - self_pose.pose.position.x )/ DelT
		twi.twist.angular.z = (yaw - yawold)/DelT
		twi.twist.linear.y = (pose.position.y - self_pose.pose.position.y )/ DelT
		twi.twist.angular.x = twi.twist.angular.y = twi.twist.linear.z = 0.0
		vel_pub.publish(twi)
		self_pose = vel
		yawold = yaw
	else:
		once = True
		self_pose = vel
		rot = [0, 0, self_pose.pose.orientation.z, self_pose.pose.orientation.w]
		(roll, pitch, yawold) = euler_from_quaternion(rot)

def odom_sub(odomData):
	twi = TwistStamped()
	twi.header = odomData.header
	twi.twist = odomData.twist.twist
	vel_pub.publish(twi)

def object_sub(objects):
	global tf_list
	count = len(objects.markers)
	if count > 0:
		marker_array_ = MarkerArray()
		targetf = "/hero" # base_link or "/zoe/base_link" 
		sourcef = "/map" # global map or odom "/zoe/odom"
		tf_list.waitForTransform (targetf, sourcef, rospy.Time(0), rospy.Duration(1))
		for i in range(count):
			pos = PoseStamped()
			pos.pose = objects.markers[i].pose
			pos.header.frame_id = sourcef
			pos.header.stamp = rospy.Time(0)
			pose_tf = tf_list.transformPose(targetf , pos)
			pose_tf.pose.position.y += 40
			x_ = pose_tf.pose.position.x 
			y_ = pose_tf.pose.position.y 
			pose_tf.pose.position.z = 1.5
			marker_= objects.markers[i]
			marker_.header.frame_id = targetf
			marker_.pose = pose_tf.pose
			marker_array_.markers.append(marker_)
		marker_pub.publish(marker_array_)
		marker_array_.markers = [0]*count

def callback_sub(obj):
	global minid
	marker_array_ = MarkerArray()
	count = len(obj.objects)
	for i in range(count):
		marker_ = Marker()
		marker_.ns = "carla_objects"
		ido = obj.objects[i].id
		if ido == minid or ido == minid+1:
			if ido == minid:
				marker_.id = 0
			if ido == minid + 1:
				marker_.id = 1
			marker_.header = obj.header
			marker_.type = 1 #cube
			marker_.action = marker_.ADD
			marker_.pose =  obj.objects[i].pose
			marker_.lifetime = rospy.Duration.from_sec(1)
			marker_.scale.x = obj.objects[i].shape.dimensions[0]
			marker_.scale.y = obj.objects[i].shape.dimensions[1]
			marker_.scale.z = obj.objects[i].shape.dimensions[2]
			marker_.color.a = 1.0
			marker_.color.g = 1.0
			marker_.text = "Car"
			marker_array_.markers.append(marker_)
	object_pub.publish(marker_array_)

############################################ section 3 main body
minid = 496 #id of the object to be tracked. Can be extracted from bag file
if __name__ == '__main__':
    try:
    	rospy.init_node('carla_objects', anonymous=True)
    	rospy.Subscriber("/carla/objects", ObjectArray, callback_sub)
    	tf_list = TransformListener()
    	rospy.Subscriber("/objects_data", MarkerArray, object_sub)
    	rospy.Subscriber("/carla/hero/odometry", Odometry, odom_sub)
    	object_pub = rospy.Publisher("/objects_data", MarkerArray, queue_size=10)
    	marker_pub = rospy.Publisher("/filter_objects_data", MarkerArray, queue_size=10)

    	vel_pub = rospy.Publisher("self_vel", TwistStamped, queue_size = 10)
    	rospy.spin()
    except rospy.ROSInterruptException:
        pass