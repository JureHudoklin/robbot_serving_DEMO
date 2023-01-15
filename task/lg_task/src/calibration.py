#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import rospy
import tf
import tf.transformations
from tf2_ros import StaticTransformBroadcaster

from config.config import Config
from device_interface.motion_planner import CupMotionPlanner, PlateMotionPlanner
from device_interface.vision_sensor_interface import VisionSensorInterfaceScale, VisionSensorInterface
from detection_interface.mobile_robot_pose_detection_client import MobileRobotPoseDetectionClient

from geometry_msgs.msg import PoseStamped, TransformStamped

from robotiq_2f_gripper_control.robotiq_2f_gripper_ctrl import RobotiqCGripper


class PoseManager(object):
    def __init__(self, config):
        assert isinstance(config, Config)
        # get config
        self.config = config

        self.tf_sbr = StaticTransformBroadcaster()
        self.tf_listener = tf.TransformListener()

    def broadcast_place_pose(self, mobile_pose):
        assert isinstance(mobile_pose, PoseStamped)

        mobile_tf = TransformStamped()
        mobile_tf.header.frame_id = mobile_pose.header.frame_id
        mobile_tf.header.stamp = rospy.Time.now()
        mobile_tf.child_frame_id = self.config.MOBILE_ID
        mobile_tf.transform.translation.x = mobile_pose.pose.position.x
        mobile_tf.transform.translation.y = mobile_pose.pose.position.y
        mobile_tf.transform.translation.z = mobile_pose.pose.position.z
        mobile_tf.transform.rotation.x = mobile_pose.pose.orientation.x
        mobile_tf.transform.rotation.y = mobile_pose.pose.orientation.y
        mobile_tf.transform.rotation.z = mobile_pose.pose.orientation.z
        mobile_tf.transform.rotation.w = mobile_pose.pose.orientation.w

        # set cup place pose
        cup_place_tf = TransformStamped()
        cup_place_tf.header.frame_id = self.config.MOBILE_ID
        cup_place_tf.header.stamp = rospy.Time.now()
        cup_place_tf.child_frame_id = self.config.CUP_PLACE_ID
        cup_place_tf.transform.translation.x = self.config.CUP_PLACE_POS[0]
        cup_place_tf.transform.translation.y = self.config.CUP_PLACE_POS[1]
        cup_place_tf.transform.translation.z = self.config.CUP_PLACE_POS[2]
        cup_place_tf.transform.rotation.x = self.config.CUP_PLACE_ORI[0]
        cup_place_tf.transform.rotation.y = self.config.CUP_PLACE_ORI[1]
        cup_place_tf.transform.rotation.z = self.config.CUP_PLACE_ORI[2]
        cup_place_tf.transform.rotation.w = self.config.CUP_PLACE_ORI[3]

        # set plate place pose( e_xyz = (-125, -0, 90))
        plate_place_tf = TransformStamped()
        plate_place_tf.header.frame_id = self.config.MOBILE_ID
        plate_place_tf.header.stamp = rospy.Time.now()
        plate_place_tf.child_frame_id = self.config.PLATE_PLACE_ID
        plate_place_tf.transform.translation.x = self.config.PLATE_PLACE_POS[0]
        plate_place_tf.transform.translation.y = self.config.PLATE_PLACE_POS[1]
        plate_place_tf.transform.translation.z = self.config.PLATE_PLACE_POS[2]
        plate_place_tf.transform.rotation.x = self.config.PLATE_PLACE_ORI[0]
        plate_place_tf.transform.rotation.y = self.config.PLATE_PLACE_ORI[1]
        plate_place_tf.transform.rotation.z = self.config.PLATE_PLACE_ORI[2]
        plate_place_tf.transform.rotation.w = self.config.PLATE_PLACE_ORI[3]

        self.tf_sbr.sendTransform([mobile_tf, cup_place_tf, plate_place_tf])
        rospy.sleep(0.2)

    def get_cup_place_pose(self):
        # get transform from /tf
        translation, rotation = self.tf_listener.lookupTransform(
            self.config.BASE_NAME,
            self.config.CUP_PLACE_ID,
            rospy.Time(0)
        )

        # return pose
        cup_place_pose = PoseStamped()
        cup_place_pose.header.frame_id = self.config.CUP_PLACE_ID
        cup_place_pose.pose.position.x = translation[0]
        cup_place_pose.pose.position.y = translation[1]
        cup_place_pose.pose.position.z = translation[2]
        cup_place_pose.pose.orientation.x = rotation[0]
        cup_place_pose.pose.orientation.y = rotation[1]
        cup_place_pose.pose.orientation.z = rotation[2]
        cup_place_pose.pose.orientation.w = rotation[3]
        return cup_place_pose

    def get_plate_place_pose(self):
        # get transform from /tf
        translation, rotation = self.tf_listener.lookupTransform(
            self.config.BASE_NAME,
            self.config.PLATE_PLACE_ID,
            rospy.Time(0)
        )

        # return pose
        plate_place_pose = PoseStamped()
        plate_place_pose.header.frame_id = self.config.PLATE_PLACE_ID
        plate_place_pose.pose.position.x = translation[0]
        plate_place_pose.pose.position.y = translation[1]
        plate_place_pose.pose.position.z = translation[2]
        plate_place_pose.pose.orientation.x = rotation[0]
        plate_place_pose.pose.orientation.y = rotation[1]
        plate_place_pose.pose.orientation.z = rotation[2]
        plate_place_pose.pose.orientation.w = rotation[3]
        return plate_place_pose

    def get_pose_from_config(self, position, orientation):
        pose = PoseStamped()

        pose.header.frame_id = self.config.BASE_NAME
        pose.header.stamp = rospy.Time(0)
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]
        return pose

    def get_cup_standby_pose(self):
        standby_pos = self.get_pose_from_config(
            self.config.CUP_STANDBY_POS,
            self.config.CUP_STANDBY_ORI,
        )
        return standby_pos

    def get_plate_standby_pose(self):
        standby_pose = self.get_pose_from_config(
            self.config.PLATE_STANDBY_POS,
            self.config.PLATE_STANDBY_ORI,
        )
        return standby_pose

    def get_cup_pick_pose(self):
        pick_pos = self.get_pose_from_config(
            self.config.CUP_PICK_POS,
            self.config.CUP_PICK_ORI,
        )
        return pick_pos

    def get_plate_pick_pose(self):
        pick_pos = self.get_pose_from_config(
            self.config.PLATE_PICK_POS,
            self.config.PLATE_PICK_ORI,
        )
        return pick_pos


class TaskManager(object):
    def __init__(self):
        self.config = Config()

    def init_motion_planner(self):
        # init motion planner
        self.cup_motion_planner = CupMotionPlanner(
            self.config.CUP_GROUP_NAME,
            self.config.BASE_NAME,
        )
        self.plate_motion_planner = PlateMotionPlanner(
            self.config.PLATE_GROUP_NAME,
            self.config.BASE_NAME,
        )
        rospy.loginfo('Inintialize motion planner')

    def init_gripper(self):
        self.robotiq_2f_gripper = RobotiqCGripper()
        self.robotiq_2f_gripper.reset()
        rospy.sleep(0.01)
        self.robotiq_2f_gripper.wait_for_connection()
        if self.robotiq_2f_gripper.is_reset():
            self.robotiq_2f_gripper.reset()
            self.robotiq_2f_gripper.activate()
        self.robotiq_2f_gripper.goto(-1, 0.1, 30, block=True) # close
        self.robotiq_2f_gripper.goto(0.8, 0.1, 30, block=True) # open
        rospy.loginfo('Inintialize gripper')

        # pre-grasp for the palte
        self.robotiq_2f_gripper.goto(0.05, 0.1, 30, block=True) # ready to grasp

    def init_pose_manager(self):
        self.pose_manager = PoseManager(self.config)
        rospy.loginfo('Inintialize pose manager')

    def serve_cup_task(self):
        self.robotiq_2f_gripper.goto(0.8, 0.1, 30, block=True)  # open

        # cup pick plan
        plan = self.cup_motion_planner.get_pick_plan(
            self.pose_manager.get_cup_pick_pose(),
            self.config.CUP_PICK_Z_OFFSET,
        )
        self.cup_motion_planner.execute_plan(
            plan,
            auto=self.config.AUTO_EXECUTE,
        )
        self.robotiq_2f_gripper.goto(-1, 0.1, 30, block=True) # close

        # cup place plan
        plan = self.cup_motion_planner.get_place_plan(
            self.pose_manager.get_cup_place_pose(),
            self.config.CUP_MIDDLE_POS,
            z_offset=self.config.CUP_PICK_Z_OFFSET,
        )
        scaled_plan = self.cup_motion_planner.scale_plan(plan, self.config.SERVE_SPEED_SCALE)
        self.cup_motion_planner.execute_plan(scaled_plan, auto=self.config.AUTO_EXECUTE)
        self.robotiq_2f_gripper.goto(0.8, 0.1, 30, block=True)  # open

        # cup back to standby plan
        plan = self.cup_motion_planner.get_standby_plan(
            pm.get_cup_standby_pose(),
            self.config.CUP_MIDDLE_POS,
            z_offset=self.config.CUP_PICK_Z_OFFSET,
        )
        scaled_plan = self.cup_motion_planner.scale_plan(plan, self.config.FAST_SPEED_SCALE)
        self.cup_motion_planner.execute_plan(scaled_plan, auto=self.config.AUTO_EXECUTE)

    def serve_plate_task(self):
        # plate pick plan
        plan = self.plate_motion_planner.get_pick_plan(
            self.pose_manager.get_plate_pick_pose(),
            self.config.PLATE_PICK_APP_OFFSET,
        )
        self.plate_motion_planner.execute_plan(plan)

        # plate place plan
        plan = self.plate_motion_planner.get_place_plan(
                    self.pose_manager.get_plate_place_pose(),
                    self.config.PLATE_MIDDLE_POS,
                    z_offset=self.config.PLATE_PICK_Z_OFFSET,
        )
        scaled_plan = self.plate_motion_planner.scale_plan(plan, 2.0)
        self.plate_motion_planner.execute_plan(scaled_plan)

        # plate back to standby plan
        plan = self.plate_motion_planner.get_standby_plan(
            self.pose_manager.get_plate_standby_pose(),
            self.config.PLATE_MIDDLE_POS,
        )
        scaled_plan = self.plate_motion_planner.scale_plan(plan, 2.0)
        self.plate_motion_planner.execute_plan(scaled_plan)

    def run(self):
        # 1) detect mobile robot
        # mobile_pose = ...
        mobile_pose = PoseStamped()
        mobile_pose.header.frame_id = config.BASE_NAME
        mobile_pose.pose.position.x = 0.45
        mobile_pose.pose.position.y = 0.45
        mobile_pose.pose.position.z = 0.220
        mobile_pose.pose.orientation.x = 0.0
        mobile_pose.pose.orientation.y = 0.0
        mobile_pose.pose.orientation.z = -0.258819
        mobile_pose.pose.orientation.w = 0.9659258

        # 2) broadcast mobile pose
        self.pose_manager.broadcast_place_pose(mobile_pose)

        # 3) do server cup task
        self.serve_cup_task()

        # 4) do server plate task
        self.serve_plate_task()


if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('get_current_pose_node')

    # get config
    config = Config()

    task_manager = TaskManager()
    task_manager.run()
