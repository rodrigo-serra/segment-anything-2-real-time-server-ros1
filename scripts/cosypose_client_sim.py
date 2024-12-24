#!/usr/bin/env python3
import rospy
import message_filters
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
        self.meshes_size = rospy.get_param("~objs_meshes_size", "mm")
        self.meshes_path = rospy.get_param("~meshes_path", os.getenv("HAPPYPOSE_DATA_DIR") + "/assets/")
        rospy.logwarn(self.meshes_path)

        # Node variables
        self.detectionResults = []
        self.rgb_img = None
        self.camera_params = None

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")
    
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
        
        pose_estimate = self.get_pose_from_server()
        rospy.logwarn(pose_estimate)
        
        if pose_estimate:
            # Send the result
            result = happypose_ros1.msg.PoseEstimateResult()
            # result.obj_pose = pose_estimate
            self._action_server.set_succeeded(result)
        else:
            rospy.logerr("Failed to get pose estimate from Happypose server.")
            self._action_server.set_aborted()

    def simulateDetectionResults(self):
        """Simulate detection results by reading the necessary data"""
        self.detectionResults = readJsonFile(self.assets_dir, self.bbox_data)        
        self.rgb_img = readImg(self.assets_dir + self.image)
        rospy.loginfo(self.detectionResults)

        # Read camera parameters from file (simulating)
        camera_data_path = Path(self.assets_dir + "/camera_data.json")
        with camera_data_path.open("r") as file:
            self.camera_params = json.load(file)

        rospy.loginfo(self.camera_params)

    def get_pose_from_server(self):
        """Fetch pose estimate from external Happypose server over HTTP (with JSON and image)."""
        try:
            # Prepare the JSON data with extra parameters
            json_data = {
                "dataset": self.dataset,
                "bbox_info": self.detectionResults,  # Ensure this is in the right format
                "objs_meshes_size": self.meshes_size,
                "meshes_path": self.meshes_path,
                "camera_params": self.camera_params  # Make sure this is correctly loaded
            }

            rospy.loginfo("Sending JSON Data: %s", json.dumps(json_data))

            # Convert image to byte array using OpenCV
            _, img_encoded = cv2.imencode('.jpg', self.rgb_img)
            img_data = io.BytesIO(img_encoded.tobytes())

            # Prepare the files to send in the request
            files = {
                'rgb_img': ('image.jpg', img_data, 'image/jpeg'),  # Make sure the key is "rgb_img"
                'data.json': ('data.json', json.dumps(json_data), 'application/json')  # JSON file
            }

            # Send HTTP POST request with both the image and JSON data
            response = requests.post(self.server_url, files=files)

            if response.status_code == 200:
                # Parse response JSON to extract pose estimate
                return response.json()  # Expected to return pose data as a dictionary
            else:
                rospy.logerr(f"Error from Happypose server: {response.status_code}")
                rospy.logerr(f"Error response: {response.text}")  # Print the response text to debug
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
