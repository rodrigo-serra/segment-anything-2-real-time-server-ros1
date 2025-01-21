#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mbot_perception_msgs.msg import RecognizedObject3D, RecognizedObject3DList
import actionlib
import os
import requests
import json
import cv2
import happypose_ros1.msg
import io
from my_robot_common.modules.common import readImg, readJsonFile, readYamlFile, transformPoseFrame, init_tf
import rospkg

class HappyposeServer():
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('happypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Initiallize TFs
        init_tf()

        # Set datadir
        rospack = rospkg.RosPack()
        self.pkg_dir = rospack.get_path('happypose_ros1')
        # self.assets_dir = self.pkg_dir + "/assets/two_objs_ycb/"
        self.assets_dir = self.pkg_dir + "/assets/cracker_box/"

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
        self.meshes_path = rospy.get_param("~meshes_path", os.getenv("CAD_DATA_DIR"))
        self.config_dir = self.meshes_path + "/"

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
        self.req_frame = None

        # Load the model at initialization
        if not self.load_model():
            rospy.logerr("Failed to load the model on the Happypose server. Exiting.")
            rospy.signal_shutdown("Failed to load model")

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting happypose Action Server")

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


    def processGoalRequest(self, info):
        """Process the list of requested objects, match them to the label map, etc."""
        # Apply obj name and label mapping according to config file. The server does not need to know obj name, only label
        # Load and validate object label map
        obj_label_map = readYamlFile(self.config_dir, self.obj_label_map_file_name)
        if not obj_label_map:
            rospy.logerr("Object label map is empty or not found.")
            self._action_server.set_aborted(text="Object label map is empty or not found.")
            return

        # Get frame from msg
        self.req_frame = info.frame

        for obj in info.obj_names:
            obj_name = obj
            official_obj_name = obj_name

            # Check if the obj_name is directly in the label map
            if obj_name not in obj_label_map:
                # If not found, check if it's one of the possible names in the label map
                matched_name = None
                for key, value in obj_label_map.items():
                    if obj_name in value['possible_names']:
                        matched_name = key
                        break
                
                if matched_name is None:
                    rospy.logerr(f"Object '{obj_name}' not found in the label map or its possible names.")
                    self._action_server.set_aborted(text=f"Object '{obj_name}' not found in the label map or its possible names.")
                    return
                
                # Use the matched obj_name from the label map
                official_obj_name = matched_name
            else:
                rospy.logerr(f"Object '{obj_name}' will not match object detector names. Please set correct name!")
                if self._action_server.is_active():  # Check if the action server is still active
                    self._action_server.set_aborted(text=f"Object '{obj_name}' will not match object detector names. Please set correct name!")
                return
            
            # Now that we have the correct obj_name, proceed with the label mapping
            obj_label = obj_label_map[official_obj_name]['label']
            
            dic = {"label": obj_label}
            # Note: Both lists below will be "linked" by their index, 
            # e.g. self.bbox_info[0] and self.obj_names[0] will have info about the same obj
            self.bbox_info.append(dic)
            self.obj_names.append(obj_name)
            rospy.logwarn(self.obj_names)


    def execute_callback(self, goal_handle):
        # 0. Applying label/dataset mapping according to request
        self.processGoalRequest(goal_handle.info)
        rospy.loginfo("Starting happypose estimation…")

        # 1. Process detection results (Simulation with image)
        self.sendFeedback(msg="Simulating cam image and detectron results")
        self.simulateDetectionResults()

        # 2. Query the external serverr
        self.sendFeedback(msg="Sending data to Happypose server")
        pose_estimate = self.get_pose_from_server()
        rospy.logwarn(pose_estimate)
        
        # 3. Covert & Return result
        if pose_estimate:
            res_msg = self.convertPoseRes2Msg(pose_estimate=pose_estimate)
            result = happypose_ros1.msg.PoseEstimateResult()
            result.obj_pose = res_msg
            self._action_server.set_succeeded(result)
        else:
            self.abort_action(error_msg="Failed to get pose estimate from Happypose server.")

        self.resetVars()

    
    def abort_action(self, error_msg: str):
        """Utility function to abort the action and log the error."""
        rospy.logerr(error_msg)
        if self._action_server.is_active():
            self._action_server.set_aborted(text=error_msg)
        return

    def sendFeedback(self, msg: str):
        """Utility function to send feedback"""
        feedback_msg = happypose_ros1.msg.PoseEstimateFeedback()
        feedback_msg.current_process = msg
        self._action_server.publish_feedback(feedback_msg)


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
                "bbox_info": self.detectionResults,
                "camera_params": self.camera_params
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
        self.req_frame = None

    
    def convertPoseRes2Msg(self, pose_estimate: str):
        if isinstance(pose_estimate, str):
            try:
                pose_estimate = json.loads(pose_estimate)
            except json.JSONDecodeError as e:
                rospy.logerr(f"Failed to deserialize pose estimate: {e}")
                raise ValueError("Pose estimate could not be deserialized.")

        if not isinstance(pose_estimate, list):
            rospy.logerr(f"Expected a list, but got: {type(pose_estimate)}")
            raise ValueError("Pose estimate should be a list of dictionaries.")

        recognized_objects_list = RecognizedObject3DList()
        recognized_objects_list.header.stamp = rospy.Time.now()
        recognized_objects_list.header.frame_id = 'azure_rgb_camera_link'
        recognized_objects_list.image_header = recognized_objects_list.header

        rospy.logwarn("Original pose frame: " + recognized_objects_list.header.frame_id)

        for obj in pose_estimate:
            recognized_object = RecognizedObject3D()
            label = obj.get("label", "unknown")

            # Match with our local arrays
            obj_name = "unknown"
            for idx, info in enumerate(self.bbox_info):
                if info.get("label") == label:
                    obj_name = self.obj_names[idx]
                    idx_for_obj = idx
                    break
            
            recognized_object.class_name = obj_name

            pose_data = obj.get("pose", [])
            # Example: pose_data = [ [qx,qy,qz,qw], [x,y,z] ]
            if len(pose_data) != 2:
                rospy.logwarn(f"Pose data missing or malformed for object: {label}")
                recognized_objects_list.objects.append(recognized_object)
                continue

            # If the user specified a 'frame', attempt transform
            if obj_name != "unknown":
                if self.req_frame:
                    # Create a PoseStamped from the current 'pose_data'
                    # We do a partial "Pose" -> "PoseStamped"
                    pose_st = PoseStamped()
                    pose_st.header = recognized_objects_list.header
                    pose_st.pose.orientation.x = pose_data[0][0]
                    pose_st.pose.orientation.y = pose_data[0][1]
                    pose_st.pose.orientation.z = pose_data[0][2]
                    pose_st.pose.orientation.w = pose_data[0][3]
                    pose_st.pose.position.x = pose_data[1][0]
                    pose_st.pose.position.y = pose_data[1][1]
                    pose_st.pose.position.z = pose_data[1][2]

                    # Attempt transform
                    try:
                        new_pose_st = transformPoseFrame(pose_st, self.req_frame)
                        if new_pose_st is not None:
                            # Overwrite recognized_objects_list header
                            recognized_objects_list.header = new_pose_st.header
                            # Overwrite pose_data with the newly transformed
                            # Convert geometry_msgs/Pose back to [orientation, position]
                            o = new_pose_st.pose.orientation
                            p = new_pose_st.pose.position
                            pose_data = [
                                [o.x, o.y, o.z, o.w],
                                [p.x, p.y, p.z]
                            ]
                        else:
                            rospy.logerr("Could not transform to requested frame: " + self.req_frame)
                    except Exception as e:
                        rospy.logerr(f"Transform to frame '{self.req_frame}' failed: {e}")

            # Now fill recognized_object with the final pose_data
            orientation = pose_data[0]
            recognized_object.pose.orientation.x = orientation[0]
            recognized_object.pose.orientation.y = orientation[1]
            recognized_object.pose.orientation.z = orientation[2]
            recognized_object.pose.orientation.w = orientation[3]

            position = pose_data[1]
            recognized_object.pose.position.x = position[0]
            recognized_object.pose.position.y = position[1]
            recognized_object.pose.position.z = position[2]

            recognized_objects_list.objects.append(recognized_object)

        return recognized_objects_list


def main(args=None):
    # Init ROS1 and give the node a name
    rospy.init_node("happypose_server")
    happypose_server = HappyposeServer()
    rospy.spin()


if __name__ == '__main__':
    main()
