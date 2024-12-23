#!/usr/bin/env python3
import rospy, message_filters
from sensor_msgs.msg import Image
from detectron2_ros.msg import RecognizedObjectWithMaskArrayStamped
from mbot_perception_msgs.msg import RecognizedObject3D
import actionlib
import os
import requests
import json
import cv2
import numpy as np
from pathlib import Path
import happypose_ros1.msg
from cv_bridge import CvBridge
import io
from my_robot_common.modules.common import readImg, readJsonFile
import rospkg

class CosyposeServer():
    # Create Feedback and Result messages
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('cosypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Set datadir
        rospack = rospkg.RosPack()
        self.pkg_dir = rospack.get_path('happypose_ros1')
        self.assets_dir = self.pkg_dir + "/assets/"

        # Load parameters
        self.server_url = rospy.get_param("~server_url", "http://localhost:5000/get_pose")
        self.dataset = rospy.get_param('~dataset', 'ycbv')
        self.image = rospy.get_param("~image_name", "image_rgb.png")
        self.bbox_data = rospy.get_param("~json_path", "object_data.json")
        
        # Node variables
        # Sync msgs vars
        self.detectionResults = None
        self.rgb_img = None

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")

    
    # Callback function to run after acknowledging a goal from the client
    def execute_callback(self, goal_handle):
        obj_name = goal_handle.obj_info.obj_name
        chosen_frame = goal_handle.obj_info.frame

        rospy.loginfo("Starting cosypose estimation…")

        # Read cam image and detectron results
        feedback_msg = happypose_ros1.msg.PoseEstimateFeedback()
        feedback_msg.current_process = "Reading cam image and detectron results"
        self._action_server.publish_feedback(feedback_msg)

        # Process detection results
        self.simulateDetectionResults()

        # Send pose estimate request to Happypose server
        feedback_msg.current_process = "Sending data to Happypose server"
        self._action_server.publish_feedback(feedback_msg)

        pose_estimate = self.get_pose_from_server(obj_name, chosen_frame)

        if pose_estimate:
            # Send the result
            result = happypose_ros1.msg.PoseEstimateResult()
            result.pose = pose_estimate
            self._action_server.set_succeeded(result)
        else:
            rospy.logerr("Failed to get pose estimate from Happypose server.")
            self._action_server.set_aborted()

    
    def simulateDetectionResults(self):
        """TODO"""
        self.detectionResults = readJsonFile(self.assets_dir, self.bbox_data)
        self.rgb_img = readImg(self.assets_dir + self.image)
        self.detectionResults = self.detectionResults[0]
        rospy.loginfo(self.detectionResults)

    
    def get_pose_from_server(self, obj_name, frame):
        """Fetch pose estimate from external Happypose server over HTTP (with JSON and image)."""
        try:
            # URL of your Happypose server
            url = f"http://localhost:5000/get_pose"

            # Convert image to byte array using OpenCV
            _, img_encoded = cv2.imencode('.jpg', self.rgb_img)
            img_data = io.BytesIO(img_encoded.tobytes())

            # Prepare the JSON data (other parameters)
            json_data = {
                "dataset": self.dataset,
                "object": obj_name,
                "frame": frame,
                "bbox_info": self.detectionResults, 
            }

            # Prepare the files to send in the request
            files = {
                'image': ('image.jpg', img_data, 'image/jpeg'),  # Image file
                'data': ('data.json', json.dumps(json_data), 'application/json')  # JSON file
            }

            # Send HTTP POST request with both the image and JSON data
            response = requests.post(url, files=files)

            if response.status_code == 200:
                # Parse response JSON to extract pose estimate
                return response.json()  # Expected to return pose data as a dictionary
            else:
                rospy.logerr(f"Error from Happypose server: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Request to Happypose server failed: {e}")
            return None

    


    
def main(args=None):
    # Init ROS1 and give the node a name
    rospy.init_node("cosypose_server")
    cosypose_server = CosyposeServer()
    rospy.spin()


if __name__ == '__main__':
    main()
