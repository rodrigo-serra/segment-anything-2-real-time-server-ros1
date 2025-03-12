#!/usr/bin/env python3
import rospy
import requests
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from detectron2_ros.msg import RecognizedObjectWithMaskArrayStamped
from cv_bridge import CvBridge
import message_filters
import tf2_ros
import geometry_msgs.msg
from centroid_computation import compute_3d_position_from_mask_and_depth

class SAM2(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Parameters
        self.cam_input_topic = rospy.get_param('~cam_input', '/azure/rgb/image_raw')
        self.cam_depth_input_topic = rospy.get_param('~cam_depth_input', '/azure/depth_to_rgb/image_raw')
        self.cam_info_topic = rospy.get_param('~cam_info', '/azure/depth_to_rgb/camera_info')
        self.detections_topic = rospy.get_param('~detectron_result', '/detectron2_ros/result')
        self.FLASK_SERVER_URL = 'http://localhost:5000/process_rgb'
        self.FLASK_SERVER_INIT_URL = 'http://localhost:5000/initialize_model'

        self.camera_params = None

        # Initialization flags
        self.initial_mask = None
        self.initialized = False
        self._queue_size = 10
        self._slop = 0.2
        self._sync_wait_time = 5.0
        self.readDetectronMsgs = False
        self.detectionResults = None

        self.new_image_frame = False
        self.rgb_img = None
        self.rgb_img_header = None
        self.depth_img = None

        self.show_live_result = True
        self.show_centroid = True

        # Fetch and store camera parameters
        self.get_camera_params()

        self.subscribeCamAndDetectron()

        # Wait for synchronized messages
        wait_time = rospy.Duration(self._sync_wait_time)
        start_time = rospy.Time.now()
        rospy.loginfo("Waiting for synchronized messages...")
        while not self.readDetectronMsgs and rospy.Time.now() - start_time < wait_time:
            rospy.sleep(0.1)

        if not self.readDetectronMsgs:
            rospy.logerr("Failed to receive synchronized messages within timeout.")
            rospy.signal_shutdown("Failed to get sync msgs")

        try:
            if self.processDetectionResults() is None:
                rospy.logerr("Failed to read mask from Detectron2 result!")
                rospy.signal_shutdown("Failed to read mask from Detectron2 result!")
        except Exception as e:
            rospy.logerr(f"Error processing detection results: {e}")
            rospy.signal_shutdown("Failed to process detection results!")

        try:
            self.initialize_model_with_mask()
        except Exception as e:
            rospy.logerr(f"Error initializing model with mask: {e}")
            rospy.signal_shutdown("Failed to initialize SAM2 model with mask!")

        # Subscribe to rgbd after initialization
        self.subscribeRGBGD()


    def get_camera_params(self):
        """Subscribe to camera info topic to fetch camera parameters."""
        try:
            rospy.loginfo("Subscribing to camera info topic to fetch camera parameters...")
            camera_info = rospy.wait_for_message(self.cam_info_topic, CameraInfo, timeout=5)
            self.camera_params = {
                "K": [
                    [camera_info.K[0], camera_info.K[1], camera_info.K[2]],
                    [camera_info.K[3], camera_info.K[4], camera_info.K[5]],
                    [camera_info.K[6], camera_info.K[7], camera_info.K[8]]
                ],
                "resolution": [camera_info.height, camera_info.width]
            }
            rospy.loginfo(f"Camera parameters fetched successfully: {self.camera_params}")
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to fetch camera parameters: {e}")
            rospy.signal_shutdown("Failed to fetch camera parameters")


    def detectronSynchronizedCallback(self, detectronMsg, imgMsg):
        """Callback for synchronized detection and camera messages."""
        rospy.loginfo("Synchronized messages received.")
        self.detectionResults = detectronMsg
        try:
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


    def subscribeCamAndDetectron(self):
        """Sets up synchronized subscribers for one-time use per goal."""
        try:
            self.readDetectronMsgs = False
            self.img_sub = message_filters.Subscriber(self.cam_input_topic, Image)
            self.detector_sub = message_filters.Subscriber(self.detections_topic, RecognizedObjectWithMaskArrayStamped)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.detector_sub, self.img_sub], queue_size=self._queue_size, slop=self._slop
            )
            self.ts.registerCallback(self.detectronSynchronizedCallback)
        except Exception as e:
            rospy.logerr(f"Failed to initialize detection results synchronization: {e}")


    def processDetectionResults(self):
        """Process detectron results and converting to request msg format"""
        detected_objs = self.detectionResults.objects.objects
        closest_person = None
        max_area = 0
        for obj in detected_objs:
            class_name = obj.class_name
            if class_name == 'person':
                # Get closest person by bbox area
                x1, y1, x2, y2 = obj.bounding_box.x_offset, obj.bounding_box.y_offset, \
                                  obj.bounding_box.x_offset + obj.bounding_box.width, \
                                  obj.bounding_box.y_offset + obj.bounding_box.height
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    closest_person = obj

        if closest_person:
            try:
                # Check the encoding of the mask and handle accordingly
                # rospy.loginfo(f"Mask encoding: {closest_person.mask.encoding}")
                # Directly use the mask data as it is already single-channel
                if closest_person.mask.encoding == '8UC1':
                    mask = self.bridge.imgmsg_to_cv2(closest_person.mask, desired_encoding='passthrough')
                    self.initial_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    rospy.logwarn(f"Unsupported mask encoding: {closest_person.mask.encoding}. Skipping this mask.")
            except Exception as e:
                rospy.logerr(f"Error while converting mask: {e}")
            return closest_person
        return None


    def initialize_model_with_mask(self):
        """Send the first RGB image and mask to the Flask server for initialization."""
        _, img_encoded = cv2.imencode('.jpg', self.rgb_img)
        img_bytes = img_encoded.tobytes()

        # Open mask image and encode to bytes
        _, mask_encoded = cv2.imencode('.png', self.initial_mask)
        mask_bytes = mask_encoded.tobytes()

        files = {
            'rgb_img': ('frame.jpg', img_bytes, 'image/jpeg'),
            'mask': ('mask.png', mask_bytes, 'image/png')
        }

        response = requests.post(self.FLASK_SERVER_INIT_URL, files=files)

        if response.status_code == 200:
            self.initialized = True
            self.rgb_img = None
            self.initial_mask = None
            rospy.loginfo("SAM2 server initialized successfully.")
        else:
            rospy.logwarn("Failed to initialize SAM2 server.")


    def subscribeRGBGD(self):
        """Sets up synchronized subscribers"""
        try:
            self.img_sub = message_filters.Subscriber(self.cam_input_topic, Image)
            self.depth_sub = message_filters.Subscriber(self.cam_depth_input_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.img_sub, self.depth_sub], queue_size=self._queue_size, slop=self._slop
            )
            self.ts.registerCallback(self.camSyncCallback)
        except Exception as e:
            rospy.logerr(f"Failed to initialize synchronization: {e}")


    def camSyncCallback(self, imgMsg, depthMsg):
        """Callback for synchronized camera/depth image messages."""
        try:
            self.rgb_img = imgMsg
            self.rgb_img_header = imgMsg.header
            self.depth_img = depthMsg
            self.new_image_frame = True
        except Exception as e:
            rospy.logerr(f"Failed to convert images: {e}")
            self.rgb_img = None
            self.depth_img = None
    

    def publish_person_transform(self, X, Y, Z):
        """Publish the 3D position as a transform."""
        t = geometry_msgs.msg.TransformStamped()
        
        # Fill in the transform details
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.rgb_img_header.frame_id
        t.child_frame_id = "person_position"

        # Set the position of the person
        t.transform.translation.x = X
        t.transform.translation.y = Y
        t.transform.translation.z = Z

        # Set no rotation (you can set this if you have rotation data)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)


    def run(self):
        """Main loop to process the RGB images."""
        while not rospy.is_shutdown():
            if self.initialized:
                if self.new_image_frame:
                    rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_img, "bgr8")
                    depth_image = self.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='32FC1')

                    _, img_encoded = cv2.imencode('.jpg', rgb_image)
                    img_bytes = img_encoded.tobytes()

                    response = requests.post(self.FLASK_SERVER_URL, files={'rgb_img': img_bytes})

                    if response.status_code == 200:
                        # Process the mask received from Flask server
                        mask = np.frombuffer(response.content, np.uint8)
                        mask = cv2.imdecode(mask, cv2.IMREAD_COLOR)

                        # Compute centroid of the person in the mask
                        cx, cy, X, Y, Z = compute_3d_position_from_mask_and_depth(depth_image=depth_image, person_mask=mask, camera_intrinsics=self.camera_params)
                        
                        if self.show_live_result:
                            if cx and cy and self.show_centroid:
                                # Draw a circle at the centroid (x, y) on the RGB img
                                centroid_color = (0, 255, 0)
                                centroid_radius = 10
                                cv2.circle(rgb_image, (int(cx), int(cy)), centroid_radius, centroid_color, -1)
                            
                            rgb_image = cv2.addWeighted(rgb_image, 1, mask, 0.5, 0)
                            cv2.imshow("Frame", rgb_image)
                            cv2.waitKey(1)


                        if cx is not None and cy is not None:
                            # rospy.loginfo(f"Person's 3D Centroid Position: X={X}, Y={Y}, Z={Z}")
                            # Broadcast the transform for the person's position
                            self.publish_person_transform(X, Y, Z)

                    else:
                        rospy.logwarn("Failed to receive mask from SAM2 server.")

                self.rgb_img = None
                self.rgb_img_header = None
                self.depth_img = None
                self.new_image_frame = False
                rgb_image = None
                rospy.sleep(0.1)
            else:
                rospy.loginfo("Model not initialized yet. Waiting for initialization.")
                rospy.sleep(0.1)

def main():
    # Initialize ROS node
    rospy.init_node('sam2_ros_node')
    node = SAM2()
    node.run()

if __name__ == '__main__':
    main()
