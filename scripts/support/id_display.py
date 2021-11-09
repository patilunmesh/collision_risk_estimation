'''
@author: Unmesh Patil
@project: Collision risk estimation at CHROMA, INRIA, France (April, 2021)
About code: Shows ids of detected objects in the RVIZ 
'''
import math, rospy, sys
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

def callback_sub(marker_data):
	marker_array_ = MarkerArray()
	count = len(marker_data.markers)
	for i in range(count):
		marker_ = Marker()
		marker_.ns = "ids"
		ido = marker_data.markers[i].id
		marker_.header = marker_data.markers[i].header
		marker_.type = 9 #text
		marker_.action = marker_.ADD
		marker_.id = ido
		marker_.pose =  marker_data.markers[i].pose
		marker_.lifetime = rospy.Duration.from_sec(1)
		marker_.scale.z = 1.0
		marker_.color.a = 1.0
		marker_.color.g = 1.0
		marker_.text = str(ido)
		marker_array_.markers.append(marker_)
	object_pub.publish(marker_array_)

if __name__ == '__main__':
    try:
    	rospy.init_node('marker_repub', anonymous=True)
    	rospy.Subscriber("/kitti/tracklet", MarkerArray, callback_sub)
    	object_pub = rospy.Publisher("/objects_id", MarkerArray, queue_size=10)
    	rospy.spin()
    except rospy.ROSInterruptException:
        pass