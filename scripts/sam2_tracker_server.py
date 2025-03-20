#!/usr/bin/env python3

from flask import Flask, request, jsonify
import rospy
from std_msgs.msg import Float64MultiArray
from threading import Thread
import signal
import sys
import geometry_msgs.msg
import tf2_ros

# Initialize Flask app
app = Flask(__name__)

# Initialize ROS node
rospy.init_node('position_server', anonymous=True)

# ROS Publisher to send data to the ROS1 node
pub = rospy.Publisher('/person_position', Float64MultiArray, queue_size=10)

# TF init
tf_broadcaster = tf2_ros.TransformBroadcaster()

@app.route('/person_position', methods=['POST'])
def receive_position():
    data = request.get_json()  # Get JSON data from client
    
    # Extract x, y, z from the received data
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')

    if x is None or y is None or z is None:
        return jsonify({"error": "Invalid data received"}), 400
    else:
        position = [x, y, z]

        # Broadcast TF
        publish_person_transform(x, y, z)

        # Publish the received position to the ROS topic
        position_msg = Float64MultiArray()
        position_msg.data = position
        pub.publish(position_msg)

        return jsonify({"status": "Position received and published"}), 200


def publish_person_transform(X, Y, Z):
    """Publish the 3D position as a transform."""
    t = geometry_msgs.msg.TransformStamped()
    
    # Fill in the transform details
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera_color_optical_frame"
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
    tf_broadcaster.sendTransform(t)


def start_flask():
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)

def start_ros():
    # Spin the ROS node to keep the subscriber active
    rospy.spin()

def signal_handler(sig, frame):
    print('Shutting down...')
    rospy.signal_shutdown('ROS Node shutdown requested')
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handler for graceful shutdown (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Start Flask and ROS in separate threads
    flask_thread = Thread(target=start_flask)
    ros_thread = Thread(target=start_ros)
    
    flask_thread.start()
    ros_thread.start()
    
    flask_thread.join()
    ros_thread.join()
