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
    rospy.loginfo("[happypose_client] Feedback received: %s", feedback.current_process)

def done_cb(state, result):
    """
    Callback once the action is done.
    state: terminal state of the goal (e.g., SUCCEEDED, ABORTED, etc.)
    result: the final result message from the action server.
    """
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("[happypose_client] Goal succeeded!")
        process_result(result=result)
    else:
        rospy.logwarn("[happypose_client] Action did not succeed (state=%d).", state)

    rospy.loginfo("[happypose_client] Done callback, state=%d", state)


def process_result(result):
    if result and result.obj_pose:
        rospy.loginfo("[happypose_client] Received recognized objects:")
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
        rospy.logwarn("[happypose_client] Result is empty or missing recognized objects.")


def active_cb():
    """
    Callback once the goal becomes active.
    """
    rospy.loginfo("[happypose_client] Goal just went active.")

def main():
    rospy.init_node("happypose_client_node", anonymous=True)

    # Create an action client for the 'PoseEstimateAction' using the action name.
    # Must match the server name from happyposeServer
    action_name = "happypose_ros"
    rospy.loginfo("[happypose_client] Waiting for action server '%s'...", action_name)
    client = actionlib.SimpleActionClient(action_name, PoseEstimateAction)
    client.wait_for_server()
    rospy.loginfo("[happypose_client] Connected to action server '%s'.", action_name)

     # Create a goal
    goal = PoseEstimateGoal()
    request_info = ObjectRequestInfo()
    request_info.obj_names = ["cheezit"]
    request_info.frame = ""
    goal.info = request_info

    rospy.loginfo("[happypose_client] Sending goal with obj_names=%s, frame=%s",
                  request_info.obj_names, request_info.frame)

    # Send the goal to the server with callbacks
    client.send_goal(
        goal,
        done_cb=done_cb,
        active_cb=active_cb,
        feedback_cb=feedback_cb
    )

    # Optionally, we can wait for the result:
    # rospy.loginfo("[happypose_client] Waiting for result...")
    # client.wait_for_result()

    # result = client.get_result()
    # if result:
    #     rospy.loginfo("[happypose_client] Final result received.")
    # else:
    #     rospy.logwarn("[happypose_client] No result received or action aborted.")

    rospy.spin()

if __name__ == "__main__":
    main()
