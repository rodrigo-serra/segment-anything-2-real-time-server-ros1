#!/usr/bin/env python
import rospy
import requests
from std_msgs.msg import String

# ROS Publisher
pub = rospy.Publisher('pose_data', String, queue_size=10)

def get_pose_from_server():
    try:
        # Send HTTP GET request to the Happypose server
        response = requests.get('http://localhost:5000/get_pose')
        if response.status_code == 200:
            return response.json()  # Pose data as JSON
        else:
            rospy.logwarn(f"Error getting pose from server: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        rospy.logwarn(f"Request failed: {e}")
        return None

def pose_publisher():
    rospy.init_node('pose_publisher', anonymous=True)
    rate = rospy.Rate(1)  # 1 Hz
    
    while not rospy.is_shutdown():
        pose = get_pose_from_server()
        if pose:
            pose_msg = str(pose)  # Convert pose data to string
            pub.publish(pose_msg)
            rospy.loginfo(f"Published pose data: {pose_msg}")
        rate.sleep()

if __name__ == '__main__':
    try:
        pose_publisher()
    except rospy.ROSInterruptException:
        pass
