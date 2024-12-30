#!/usr/bin/env python3
import rospy, message_filters
from sensor_msgs.msg import Image, CameraInfo
from detectron2_ros.msg import RecognizedObjectWithMaskArrayStamped
from mbot_perception_msgs.msg import RecognizedObject3D, RecognizedObject3DList
import actionlib
import os, json, requests, io
import cv2
import numpy as np
from pathlib import Path
import happypose_ros1.msg
from cv_bridge import CvBridge
from my_robot_common.modules.common import readYamlFile
import rospkg

class CosyposeServer():
    # Create Feedback and Result messages
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('cosypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Set datadir
        rospack = rospkg.RosPack()
        self.pkg_dir = rospack.get_path('happypose_ros1')
        self.config_dir = self.pkg_dir + "/config/"

        self.bridge = CvBridge()

        # Load parameters
        self.server_url_base = rospy.get_param("~server_url", "http://localhost:5000")
        self.load_model_url = f"{self.server_url_base}/load_model"
        self.get_pose_url = f"{self.server_url_base}/get_pose"
        self.dataset = rospy.get_param('~dataset', 'ycbv')
        self.obj_label_map_file_name = rospy.get_param('~obj_label_map_file', 'label_map.json')
        self.cam_input_topic = rospy.get_param('~cam_input', '/azure/rgb/image_raw')
        self.cam_info_topic = rospy.get_param('~cam_info', '/azure/rgb/camera_info')
        self.detections_topic = rospy.get_param('~detectron_result', '/detectron2_ros_specific/result')
        self.meshes_size = rospy.get_param("~objs_meshes_size", "mm")
        self.meshes_path = rospy.get_param("~meshes_path", os.getenv("HAPPYPOSE_DATA_DIR") + "/assets/")

        # Validate meshs path
        if not os.path.exists(self.meshes_path):
            rospy.logerr(f"Meshes path {self.meshes_path} does not exist.")
            rospy.signal_shutdown("Invalid meshes path")

        # Node variables
        # Server response time limit
        self.server_timeout_res = 10
        # Camera info timeout
        self.read_cam_info_timeout = 5
        # Sync msgs vars
        self._queue_size = 10
        self._slop = 0.2
        self._sync_wait_time = 5.0
        self.detectionResults = None
        self.rgb_img = None
        self.rgb_img_header = None
        self.readDetectronMsgs = False
        
        self.camera_params = None
        self.bbox_info = []
        self.obj_names = []

        # Fetch and store camera parameters
        self.get_camera_params()

        # Load the model at initialization
        if not self.load_model():
            rospy.logerr("Failed to load the model on the Happypose server. Exiting.")
            rospy.signal_shutdown("Failed to load model")

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")


    def get_camera_params(self):
        """Subscribe to camera info topic to fetch camera parameters."""
        try:
            rospy.loginfo("Subscribing to camera info topic to fetch camera parameters...")
            camera_info = rospy.wait_for_message(self.cam_info_topic, CameraInfo, timeout=self.read_cam_info_timeout)
            self.camera_params = {
                "K": [
                    [
                        camera_info.K[0], camera_info.K[1], camera_info.K[2]
                    ],
                    [
                        camera_info.K[3], camera_info.K[4], camera_info.K[5]
                    ],
                    [
                        camera_info.K[6], camera_info.K[7], camera_info.K[8]
                    ]
                ],
                "resolution": [
                    camera_info.height,
                    camera_info.width
                ]
            }
            rospy.loginfo(f"Camera parameters fetched successfully: {self.camera_params}")
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to fetch camera parameters: {e}")
            rospy.signal_shutdown("Failed to fetch camera parameters")
    

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
    
    
    # Callback function to run after acknowledging a goal from the client
    def execute_callback(self, goal_handle):
        # Apply obj name and label mapping according to config file. The server does not need to know obj name, only label
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
            # Note: Both lists below will be "linked" by their index, 
            # e.g. self.bbox_info[0] and self.obj_names[0] will have info about the same obj
            self.bbox_info.append(dic)
            self.obj_names.append(obj_name)

        rospy.loginfo("Starting cosypose estimation…")
        
        #########################################
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

        #########################################
        # Send pose estimate request to Happypose server
        feedback_msg.current_process = "Sending data to Happypose server"
        self._action_server.publish_feedback(feedback_msg)
        
        pose_estimate = self.get_pose_from_server()
        rospy.logwarn(pose_estimate)

        #########################################
        # TODO: Apply tf conversion according to requested frame
        
        #########################################
        # Convert results to result msg (RecognizedObject3DList)
        res_msg = self.convertPoseRes2Msg(pose_estimate=pose_estimate)

        #########################################
        # Return result
        if pose_estimate:
            # Send the result
            result = happypose_ros1.msg.PoseEstimateResult()
            result.obj_pose = res_msg
            self._action_server.set_succeeded(result)
            self.resetVars()
        else:
            rospy.logerr("Failed to get pose estimate from Happypose server.")
            self._action_server.set_aborted()

    
    def convertPoseRes2Msg(self, pose_estimate):
        # If pose_estimate is a string, try to deserialize it
        if isinstance(pose_estimate, str):
            try:
                pose_estimate = json.loads(pose_estimate)
            except json.JSONDecodeError as e:
                rospy.logerr(f"Failed to deserialize pose estimate: {e}")
                raise ValueError("Pose estimate could not be deserialized.")

        # Ensure pose_estimate is a list of dictionaries after deserialization
        if not isinstance(pose_estimate, list):
            rospy.logerr(f"Expected a list, but got: {type(pose_estimate)}")
            raise ValueError("Pose estimate should be a list of dictionaries.")
    
        # Initialize the RecognizedObject3DList message
        recognized_objects_list = RecognizedObject3DList()
        recognized_objects_list.header.stamp = rospy.Time.now()
        recognized_objects_list.header.frame_id = self.rgb_img_header.frame_id
        recognized_objects_list.image_header = self.rgb_img_header

        # Process each object in the pose_estimate
        for obj in pose_estimate:
            recognized_object = RecognizedObject3D()

            # Extract label from pose estimate
            label = obj.get("label", "unknown")
            
            # Map label to object name using bbox_info and obj_names
            obj_name = "unknown"
            for idx, info in enumerate(self.bbox_info):
                if info.get("label") == label:
                    obj_name = self.obj_names[idx]
                    break

            # Assign the mapped object name to class_name
            recognized_object.class_name = obj_name

            # Extract pose
            pose_data = obj.get("pose", [])
            if len(pose_data) == 2:
                # Extract orientation (quaternion)
                orientation = pose_data[0]
                recognized_object.pose.orientation.x = orientation[0]
                recognized_object.pose.orientation.y = orientation[1]
                recognized_object.pose.orientation.z = orientation[2]
                recognized_object.pose.orientation.w = orientation[3]

                # Extract position
                position = pose_data[1]
                recognized_object.pose.position.x = position[0]
                recognized_object.pose.position.y = position[1]
                recognized_object.pose.position.z = position[2]
            else:
                rospy.logwarn("Pose data missing or malformed for object: {}".format(obj.get("label", "unknown")))

            # Add the object to the list
            recognized_objects_list.objects.append(recognized_object)
    
        return recognized_objects_list


    def processDetectionResults(self):
        """Process detectron results and converting to request msg format"""
        detected_objs = self.detectionResults.objects.objects
        for i in range(len(detected_objs)):
            # Getting bounding box limits
            class_name = detected_objs[i].class_name
            if class_name in self.obj_names:
                # Find the index of the class name in obj_names
                class_index = self.obj_names.index(class_name)

                x_offset = detected_objs[i].bounding_box.x_offset
                width = detected_objs[i].bounding_box.width
                y_offset = detected_objs[i].bounding_box.y_offset
                height = detected_objs[i].bounding_box.height
        
                top_left = (int(x_offset), int(y_offset))
                bottom_right = (int(x_offset + width), int(y_offset + height))

                self.bbox_info[class_index]["bbox_modal"] = [
                    top_left[0],
                    top_left[1],
                    bottom_right[0],
                    bottom_right[1]
                ]


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
        self.detectionResults = detectronMsg
        try:
             # Save the header from the camera image
            self.rgb_img_header = imgMsg.header
            # Convert ROS Image message to OpenCV format
            self.rgb_img = self.bridge.imgmsg_to_cv2(imgMsg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            self.rgb_img = None

        self.readDetectronMsgs = True

        # Unsubscribe after receiving messages
        self.img_sub.unregister()
        self.detector_sub.unregister()
        self.ts = None

    
    def get_pose_from_server(self):
        """Fetch pose estimate from external Happypose server over HTTP (with JSON and image)."""
        try:
            # Prepare the JSON data for inference
            json_data = {
                "bbox_info": self.bbox_info,  # Ensure this is in the right format
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
        self.rgb_img_header = None


    
def main(args=None):
    # Init ROS1 and give the node a name
    rospy.init_node("cosypose_server")
    cosypose_server = CosyposeServer()
    rospy.spin()


if __name__ == '__main__':
    main()
