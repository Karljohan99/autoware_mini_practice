#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
rate_param = rospy.get_param('~rate', '2')
rate = rospy.Rate(int(rate_param))
message = rospy.get_param('~message', 'Hello World!')
pub = rospy.Publisher('/message', String, queue_size=10)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()
