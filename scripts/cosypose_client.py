#!/usr/bin/env python3

import rospy
import actionlib

# Import the generated action messages
from happypose_ros1.msg import (
    PoseEstimateAction,
    PoseEstimateGoal,
    PoseEstimateFeedback,
    PoseEstimateResult,
    ObjectRequestInfo
)

def feedback_cb(feedback):
    """
    Callback to receive feedback from the action server.
    """
    rospy.loginfo("[cosypose_client] Feedback received: %s", feedback.current_process)

def done_cb(state, result):
    """
    Callback once the action is done.
    state: terminal state of the goal (e.g., SUCCEEDED, ABORTED, etc.)
    result: the final result message from the action server.
    """
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("[cosypose_client] Goal succeeded!")
        process_result(result=result)
    else:
        rospy.logwarn("[cosypose_client] Action did not succeed (state=%d).", state)

    rospy.loginfo("[cosypose_client] Done callback, state=%d", state)


def process_result(result):
    if result and result.obj_pose:
        rospy.loginfo("[cosypose_client] Received recognized objects:")
        for idx, obj in enumerate(result.obj_pose.objects):
            rospy.loginfo(" - Object %d: class_name=%s, position=(%.3f, %.3f, %.3f), orientation=(%.3f, %.3f, %.3f, %.3f)",
                            idx,
                            obj.class_name,
                            obj.pose.position.x,
                            obj.pose.position.y,
                            obj.pose.position.z,
                            obj.pose.orientation.x,
                            obj.pose.orientation.y,
                            obj.pose.orientation.z,
                            obj.pose.orientation.w
            )
    else:
        rospy.logwarn("[cosypose_client] Result is empty or missing recognized objects.")


def active_cb():
    """
    Callback once the goal becomes active.
    """
    rospy.loginfo("[cosypose_client] Goal just went active.")

def main():
    rospy.init_node("cosypose_client_node", anonymous=True)

    # Create an action client for the 'PoseEstimateAction' using the action name.
    # Must match the server name from CosyposeServer
    action_name = "cosypose_ros"
    rospy.loginfo("[cosypose_client] Waiting for action server '%s'...", action_name)
    client = actionlib.SimpleActionClient(action_name, PoseEstimateAction)
    client.wait_for_server()
    rospy.loginfo("[cosypose_client] Connected to action server '%s'.", action_name)

    # Create a goal to send to the action server
    goal = PoseEstimateGoal()
    request_object = ObjectRequestInfo()
    request_object.obj_name = "cheezit"
    request_object.frame = ""
    goal.objs.append(request_object)

    rospy.loginfo("[cosypose_client] Sending goal with object request: %s", request_object.obj_name)

    # Send the goal to the server with callbacks
    client.send_goal(
        goal,
        done_cb=done_cb,
        active_cb=active_cb,
        feedback_cb=feedback_cb
    )

    # Optionally, we can wait for the result:
    # rospy.loginfo("[cosypose_client] Waiting for result...")
    # client.wait_for_result()

    # result = client.get_result()
    # if result:
    #     rospy.loginfo("[cosypose_client] Final result received.")
    # else:
    #     rospy.logwarn("[cosypose_client] No result received or action aborted.")

    rospy.spin()

if __name__ == "__main__":
    main()
