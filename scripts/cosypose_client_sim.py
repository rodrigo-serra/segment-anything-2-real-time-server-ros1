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
from my_robot_common.modules.common import readImg, readJsonFile, readYamlFile
import rospkg

class CosyposeServer():
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('cosypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Set datadir
        rospack = rospkg.RosPack()
        self.pkg_dir = rospack.get_path('happypose_ros1')
        self.assets_dir = self.pkg_dir + "/assets/two_objs_ycb/"
        self.config_dir = self.pkg_dir + "/config/"

        # Load parameters
        self.server_url_base = rospy.get_param("~server_url", "http://localhost:5000")
        self.load_model_url = f"{self.server_url_base}/load_model"
        self.get_pose_url = f"{self.server_url_base}/get_pose"
        self.dataset = rospy.get_param('~dataset', 'ycbv')
        self.obj_label_map_file_name = rospy.get_param('~obj_label_map_file', 'label_map.json')
        self.image = rospy.get_param("~image_name", "image_rgb.png")
        self.bbox_data = rospy.get_param("~json_path", "object_data.json")
        self.cam_data = rospy.get_param("~json_path", "camera_data.json")
        self.meshes_size = rospy.get_param("~objs_meshes_size", "mm")
        self.meshes_path = rospy.get_param("~meshes_path", os.getenv("HAPPYPOSE_DATA_DIR") + "/assets/")

        # Validate meshs path
        if not os.path.exists(self.meshes_path):
            rospy.logerr(f"Meshes path {self.meshes_path} does not exist.")
            rospy.signal_shutdown("Invalid meshes path")

        # Node variables
        # Server response time limit
        self.server_timeout_res = 10

        self.detectionResults = []
        self.rgb_img = None
        self.camera_params = None

        self.bbox_info = []
        self.obj_names = []

        # Load the model at initialization
        if not self.load_model():
            rospy.logerr("Failed to load the model on the Happypose server. Exiting.")
            rospy.signal_shutdown("Failed to load model")

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")

    def load_model(self):
        """Send a request to load the model on the Happypose server."""
        try:
            # Prepare JSON data for loading the model
            load_model_data = {
                "dataset": self.dataset,
                "objs_meshes_size": self.meshes_size,
                "meshes_path": self.meshes_path
            }

            # Send POST request to load the model
            response = requests.post(self.load_model_url, json=load_model_data, timeout=self.server_timeout_res)

            if response.status_code == 200:
                rospy.loginfo("Model loaded successfully on the Happypose server.")
                return True
            else:
                rospy.logerr(f"Error loading model: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Request to load the model failed: {e}")
            return False

    def execute_callback(self, goal_handle):
        # TODO: Apply obj name and label mapping according to config file. The server does not need to know obj name, only label
        # Load and validate object label map
        obj_label_map = readYamlFile(self.config_dir, self.obj_label_map_file_name)
        if not obj_label_map:
            rospy.logerr("Object label map is empty or not found.")
            self._action_server.set_aborted(text="Object label map is empty or not found.")
            return
        
        objs = goal_handle.objs
        for i in range(len(objs)): 
            obj_name = objs[i].obj_name
            if obj_name not in obj_label_map:
                rospy.logerr(f"Object '{obj_name}' not found in the label map.")
                self._action_server.set_aborted(text=f"Object '{obj_name}' not found in the label map.")
                return
            
            obj_label = obj_label_map[obj_name]
            dic = {"label": obj_label}
            self.bbox_info.append(dic)
            self.obj_names.append(obj_name)
        
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
        
        # TODO: Apply tf conversion according to requested frame

        # TODO: Convert results to result msg (RecognizedObject3DList)

        if pose_estimate:
            # Send the result
            result = happypose_ros1.msg.PoseEstimateResult()
            # result.obj_pose = pose_estimate
            self._action_server.set_succeeded(result)
            self.resetVars()
        else:
            rospy.logerr("Failed to get pose estimate from Happypose server.")
            self._action_server.set_aborted()


    def simulateDetectionResults(self):
        """Simulate detection results by reading the necessary data."""
        self.detectionResults = readJsonFile(self.assets_dir, self.bbox_data)        
        self.rgb_img = readImg(self.assets_dir + self.image)
        self.camera_params = readJsonFile(self.assets_dir, self.cam_data)


    def get_pose_from_server(self):
        """Fetch pose estimate from external Happypose server over HTTP (with JSON and image)."""
        try:
            # Prepare the JSON data for inference
            json_data = {
                "bbox_info": self.detectionResults,  # Ensure this is in the right format
                "camera_params": self.camera_params  # Make sure this is correctly loaded
            }

            rospy.loginfo("Sending JSON Data: %s", json.dumps(json_data))

            # Convert image to byte array using OpenCV
            _, img_encoded = cv2.imencode('.jpg', self.rgb_img)
            img_data = io.BytesIO(img_encoded.tobytes())

            # Prepare the files to send in the request
            files = {
                'rgb_img': ('image.jpg', img_data, 'image/jpeg'),  # Image file
                'data.json': ('data.json', json.dumps(json_data), 'application/json')  # JSON file
            }

            # Send HTTP POST request for inference
            response = requests.post(self.get_pose_url, files=files, timeout=self.server_timeout_res)

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
    

    def resetVars(self):
        rospy.loginfo("Reseting variables...")
        self.bbox_info = []
        self.obj_names = []


def main(args=None):
    # Init ROS1 and give the node a name
    rospy.init_node("cosypose_server")
    cosypose_server = CosyposeServer()
    rospy.spin()


if __name__ == '__main__':
    main()
