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

class CosyposeServer():
    # Create Feedback and Result messages
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('cosypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Load parameters
        self.dataset = rospy.get_param('~dataset', 'ycbv')
        self.cam_input_topic = rospy.get_param('~cam_input', '/azure/rgb/image_raw')
        self.cam_info_topic = rospy.get_param('~cam_info', '/azure/rgb/camera_info')
        self.detections_topic = rospy.get_param('~detectron_result', '/detectron2_ros_specific/result')

        # Node variables
        # Sync msgs vars
        self._queue_size = 10
        self._slop = 0.2
        self._sync_wait_time = 5.0
        self.detectionMsg = None
        self.rgb_img = None
        self.readDetectronMsgs = False

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")

    
    # Callback function to run after acknowledging a goal from the client
    def execute_callback(self, goal_handle):
        obj_name = goal_handle.obj_name
        chosen_frame = goal_handle.frame

        rospy.loginfo("Starting cosypose estimation…")

        # Read cam image and detectron results
        feedback_msg = happypose_ros1.msg.PoseEstimateFeedback()
        feedback_msg.current_process = "Reading cam image and detectron results"
        self._action_server.publish_feedback(feedback_msg)

        self.getDetectionResults()
        # Wait for synchronized messages
        wait_time = rospy.Duration(self._sync_wait_time)
        start_time = rospy.Time.now()
        rospy.logdebug("Waiting for synchronized messages...")
        while not self.readDetectronMsgs and rospy.Time.now() - start_time < wait_time:
            rospy.sleep(0.1)

        if not self.readDetectronMsgs:
            rospy.logerr("Failed to receive synchronized messages within timeout.")
            feedback_msg.current_process = "Failed to receive synchronized messages within timeout"
            self._action_server.publish_feedback(feedback_msg)
            self._action_server.set_aborted()
            return

        rospy.loginfo("Synchronized messages received successfully.")

        # Process detection results
        self.processDetectionResults()

        # Send pose estimate request to Happypose server
        feedback_msg.current_process = "Sending data to Happypose server"
        self._action_server.publish_feedback(feedback_msg)

        pose_estimate = self.get_pose_from_server(obj_name, chosen_frame, self.rgb_img)

        if pose_estimate:
            # Send the result
            result = happypose_ros1.msg.PoseEstimateResult()
            result.pose = pose_estimate
            self._action_server.set_succeeded(result)
        else:
            rospy.logerr("Failed to get pose estimate from Happypose server.")
            self._action_server.set_aborted()

    
    def processDetectionResults(self):
        """Sets up synchronized subscribers for one-time use per goal."""


    def getDetectionResults(self):
        """Sets up synchronized subscribers for one-time use per goal."""
        try:
            # Reset the flag for each goal
            self.readDetectronMsgs = False
            self.img_sub = message_filters.Subscriber(self.cam_input_topic, Image)
            self.detector_sub = message_filters.Subscriber(self.detections_topic, RecognizedObjectWithMaskArrayStamped)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.detector_sub, self.img_sub], queue_size=self._queue_size, slop=self._slop
            )
            self.ts.registerCallback(self.detectronSynchronizedCallback)
        except Exception as e:
            rospy.logerr(f"Failed to initialize detection results synchronization: {e}")
            self._action_server.set_aborted()

    def detectronSynchronizedCallback(self, detectronMsg, imgMsg):
        """Callback for synchronized detection and camera messages."""
        rospy.loginfo("Synchronized messages received.")
        self.detectionMsg = detectronMsg
        self.rgb_img = imgMsg
        self.readDetectronMsgs = True

        # Unsubscribe after receiving messages
        self.img_sub.unregister()
        self.detector_sub.unregister()
        self.ts = None  

    def get_pose_from_server(self, obj_name, frame, img_msg):
        """Fetch pose estimate from external Happypose server over HTTP (with JSON and image)."""
        try:
            # URL of your Happypose server
            url = f"http://localhost:5000/get_pose"
            params = {"object": obj_name, "frame": frame}

            # Convert image to byte array using OpenCV
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            _, img_encoded = cv2.imencode('.jpg', cv_image)
            img_data = io.BytesIO(img_encoded.tobytes())

            # Prepare the JSON data (other parameters)
            json_data = {
                "object": obj_name,
                "frame": frame,
                "dataset": self.dataset,
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
