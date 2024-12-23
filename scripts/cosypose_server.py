#!/usr/bin/env python3
import rospy, message_filters
from sensor_msgs.msg import Image
from detectron2_ros.msg import RecognizedObjectWithMaskArrayStamped
from mbot_perception_msgs.msg import RecognizedObject3D
import actionlib
import os
from pathlib import Path
import happypose_ros1.msg

class CosyposeServer():
    # Create Feedback and Result messages
    def __init__(self):
        # Create the server
        self._action_server = actionlib.SimpleActionServer('cosypose_ros', happypose_ros1.msg.PoseEstimateAction, self.execute_callback, False)

        # Load parameters
        self.desired_conda_env = rospy.get_param('~conda_env_name', 'happypose')
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

        # Check Happypose data dir
        data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
        if not data_dir:
            rospy.logerr("HAPPYPOSE_DATA_DIR environment variable not set.")
            rospy.signal_shutdown("Environment variable not set. Shutting down.")
            return

        example_dir = Path(data_dir) / self.dataset
        if not example_dir.exists():
            rospy.logerr(f"Example directory '{example_dir}' does not exist. Please follow download instructions.")
            rospy.signal_shutdown(f"Directory '{example_dir}' does not exist. Shutting down.")
            return
        
        rospy.loginfo(f"Found dataset in {example_dir}")

        # Check if happypose conda env is activated
        conda_env = os.getenv('CONDA_DEFAULT_ENV')
        if not conda_env:
            rospy.logerr("No conda environment is activated.")
            rospy.signal_shutdown("No conda environment is activated. Shutting down.")
            return

        if conda_env != self.desired_conda_env:
            rospy.logerr(f"Activated conda environment is '{conda_env}', but expected '{self.desired_conda_env}'.")
            rospy.signal_shutdown(f"Wrong conda environment. Expected '{self.desired_conda_env}', shutting down.")
            return

        rospy.loginfo(f"Conda environment '{conda_env}' is correctly activated.")

        # Start the server
        self._action_server.start()
        rospy.loginfo("Starting Cosypose Action Server")

    
    # Callback function to run after acknowledging a goal from the client
    def execute_callback(self, goal_handle):
        # TODO change goal action msg
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

        # Read cam parameters
        feedback_msg.current_process = "Reading cam parameters"
        self._action_server.publish_feedback(feedback_msg)


        rospy.loginfo("Finnished reading cam parameters")

        # Load models

        # Run Inference

        # Convert pose estimate

        # Send result
        # pose = RecognizedObject3D()
        result = happypose_ros1.msg.PoseEstimateResult()
        self._action_server.set_succeeded(result)

    
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

        

    
def main(args=None):
    # Init ROS1 and give the node a name
    rospy.init_node("cosypose_server")
    cosypose_server = CosyposeServer()
    rospy.spin()


if __name__ == '__main__':
    main()
