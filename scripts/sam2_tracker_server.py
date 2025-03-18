#!/usr/bin/env python3

from flask import Flask, request, jsonify
import rospy
from std_msgs.msg import Float64MultiArray
from threading import Thread

# Initialize Flask app
app = Flask(__name__)

# Initialize ROS node
rospy.init_node('position_server', anonymous=True)

# ROS Publisher to send data to the ROS1 node
pub = rospy.Publisher('/person_position', Float64MultiArray, queue_size=10)

@app.route('/person_position', methods=['POST'])
def receive_position():
    data = request.get_json()  # Get JSON data from client
    
    # Extract x, y, z from the received data
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')

    if x is None or y is None or z is None:
        return jsonify({"error": "Invalid data received"}), 400
    
    position = [x, y, z]
    
    # Publish the received position to the ROS topic
    position_msg = Float64MultiArray()
    position_msg.data = position
    pub.publish(position_msg)
    
    rospy.loginfo(f"Received and published position: X: {position[0]}, Y: {position[1]}, Z: {position[2]}")
    
    return jsonify({"status": "Position received and published"}), 200

def start_flask():
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)

def start_ros():
    # Spin the ROS node to keep the subscriber active
    rospy.spin()

if __name__ == '__main__':
    # Start Flask and ROS in separate threads
    flask_thread = Thread(target=start_flask)
    ros_thread = Thread(target=start_ros)
    
    flask_thread.start()
    ros_thread.start()
    
    flask_thread.join()
    ros_thread.join()
