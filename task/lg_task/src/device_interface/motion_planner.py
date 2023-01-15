#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import copy

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

import rospy
import rospkg
import tf
import tf.transformations
import moveit_commander
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from moveit_msgs.msg import RobotState, RobotTrajectory, DisplayTrajectory, JointConstraint, Constraints, PositionIKRequest
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneResponse, GetPositionFK, GetPositionIK

import matplotlib.pyplot as plt


class MotionPlanner(object):
    def __init__(self, group_name, pose_reference_frame):
        """Initialize MotionPlanner.

        Args:
            group_name (`str`): MoveIt! group name.
            pose_reference_frame_id (`str`): Pose reference frame id.
        """
        # get init variables
        self.group_name = group_name
        self.pose_reference_frame = pose_reference_frame

        # init tf
        self.tf = tf.TransformListener()

        # init move group
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_pose_reference_frame(self.pose_reference_frame)

        # publishers
        self.display_planned_path_pub = rospy.Publisher(
            '/move_group/display_planned_path',
            DisplayTrajectory,
            queue_size=2)
        self.debug_pose_array_pub = rospy.Publisher(
            '/debug_pose_array',
            PoseArray,
            queue_size=2,
        )

        # init services
        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        self.fk_srv.wait_for_service()
        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.ik_srv.wait_for_service()

        # sleep
        rospy.sleep(1)

        # !IMPORTANT: When we use more then multiple arms and gripper,
        # getCurrentPose sometimes returns an incorrect value.
        # https://github.com/ros-planning/moveit/issues/2715
        self.get_current_pose()

    def get_current_pose(self, eef_link=None, verbose=False):
        """Wrapper for MoveGroupCommander.get_current_pose()

        Even the pose reference frame has been set, the MoveGroupCommander.get_current_pose()
        does not return a current pose correctly. Here we enforce to transform the
        current pose from the world frame to the reference frame.

        Returns:
            `geometry_msgs/PoseStamped`: Current end-effector pose on the pose reference frame.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)
        cur_pose = self.tf.transformPose(self.pose_reference_frame, self.move_group.get_current_pose())
        if verbose:
            rospy.loginfo('End effector link: {}\n Current pose: \n{}.'.format(
                self.move_group.get_end_effector_link(),
                cur_pose))
        return cur_pose

    def display_planned_path(self, plan):
        """Display planned path in RViz.

        Args:
            plan (RobotTrajectory): A motion plan to be displayed.
        """   
        # assert type
        assert isinstance(plan, (list, RobotTrajectory))

        # generate msg
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = self.group_name
        display_trajectory.trajectory_start = self.move_group.get_current_state()

        # check the type and put correct trajectory
        if isinstance(plan, list):
            for p in plan:
                assert isinstance(p, RobotTrajectory)
            display_trajectory.trajectory = plan
        elif isinstance(plan, RobotTrajectory):
            display_trajectory.trajectory.append(plan)

        # publish the msg
        self.display_planned_path_pub.publish(display_trajectory)

    def is_pose_reachable(self, pose, eef_link=None, verbose=False):
        """Check if the pose is reachable.

        Args:
            pose (`geometry_msgs/PoseStamped`): Pose to be checked.
            eef_link (`str`): End effector link. If None, use the default end effector link.
            verbose (`bool`): Verbose mode. Default is False.

        Returns:
            `bool`: True if the pose is reachable.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)

        # check if the pose is reachable
        ik = self.get_inverse_kinematics(pose, timeout=0.05)
        if ik is None:
            rospy.logwarn('The pose is not reachable.')
            return False
        else:
            if verbose:
                rospy.loginfo('The pose is reachable.')
            return True

    def get_plan(self, pose, eef_link=None, verbose=False):
        """Get a motion plan.

        Args:
            pose (`geometry_msgs/PoseStamped`): A pose target.
            eef_link (`str`, optional): End effector link. Defaults to None.
            verbose (`bool`, optional): If True, print the plan. Defaults to False.

        Returns:
            `RobotTrajectory`: A motion plan.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)
        plan = self.move_group.plan(pose)
        if verbose:
            rospy.loginfo('Plan: {}\n End effector link: {}\n Target pose: \n{}'.format(
                plan,
                self.move_group.get_end_effector_link(),
                self.get_current_pose()))
        return plan

    def get_named_target_plan(self, target_name, eef_link=None, verbose=False):
        """Get a motion plan.

        Args:
            target_name (`str`): A named target.
            eef_link (`str`, optional): End effector link. Defaults to None.
            verbose (`bool`, optional): If True, print the plan. Defaults to False.

        Returns:
            `RobotTrajectory`: A motion plan.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)

        self.move_group.clear_pose_targets()
        self.move_group.set_named_target(target_name)
        plan = self.move_group.plan()
        if verbose:
            rospy.loginfo('Plan: {}\n End effector link: {}\n Target pose: \n{}'.format(
                plan,
                self.move_group.get_end_effector_link(),
                self.get_current_pose()))
        return plan

    def get_forward_kinematics(self, angles, eef_link=None, verbose=False):
        """Get the forward kinematics.

        Args:
            angles (`list`): A list of joint angles in rad.
            eef_link (`str`, optional): End effector link. Defaults to None.
            verbose (`bool`, optional): If True, print the plan. Defaults to False.

        Returns:
            `PoseStmaped`: A forward kinematics.
        """
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)

        # get request arguments
        header = Header()
        header.frame_id = self.pose_reference_frame
        header.stamp = rospy.Time.now()
        fk_link_names = [self.move_group.get_end_effector_link()]
        robot_state = self.move_group.get_current_state()
        robot_state.joint_state.header
        robot_state.joint_state.position = angles

        # get forward kinematics
        try:
            resp = self.fk_srv(header, fk_link_names, robot_state)
            return resp.pose_stamped[0]
        except rospy.ServiceException as e:
            rospy.logerr('Forward kinematic service call failed: {}'.format(e))
            return None

    def get_inverse_kinematics(
            self,
            pose,
            init_state=None,
            contraints=None,
            avoid_collisions=True,
            timeout=0.5,
            eef_link=None):
        """Get the inverse kinematics.

        Args:
            pose (`geometry_msgs/PoseStamped`): A pose target.
            init_state (`moveit_msgs/RobotState`, optional): A robot state. If None, use the current state. Defaults to None.
            contraints (`moveit_msgs/Constraints`, optional): A constraint. If None, use the default constraint. Defaults to None.
            avoid_collisions (`bool`, optional): If True, avoid collisions. Defaults to True.
            timeout (`float`, optional): Timeout in sec. Defaults to 0.5.
            eef_link (`str`, optional): End effector link. Defaults to None.

        Returns:
            `RobotState`: A robot state.
        """
        # check arguments
        if eef_link is not None:
            assert isinstance(eef_link, str), 'eef_link should be a string'
            self.move_group.set_end_effector_link(eef_link)
        assert isinstance(pose, PoseStamped), 'pose should be a PoseStamped'
        if init_state is None:
            init_state = self.move_group.get_current_state()
        else:
            assert isinstance(init_state, RobotState), 'init_state should be a RobotState'
        if contraints is None:
            contraints = self.move_group.get_path_constraints()
        else:
            assert isinstance(contraints, Constraints), 'contraints should be a Constraints'

        # get request arguments
        ik_request = PositionIKRequest()
        ik_request.group_name = self.group_name
        ik_request.robot_state = init_state
        ik_request.constraints = contraints
        ik_request.avoid_collisions = True
        ik_request.ik_link_name = self.move_group.get_end_effector_link()
        ik_request.pose_stamped = pose
        ik_request.timeout = rospy.Duration(timeout)

        # get inverse kinematics
        try:
            resp = self.ik_srv(ik_request)
            if resp.error_code.val == 1:
                return resp.solution
            else:
                rospy.logerr('Inverse kinematics failed with error code: {}'.format(resp.error_code.val))
                return None
        except rospy.ServiceException as e:
            rospy.logerr('Invert kinematic service call failed: {}'.format(e))
            return None

    def execute_plan(self, plan, wait=True, auto=False):
        """Execute a motion plan.

        Args:
            plan (`RobotTrajectory`): A motion plan to be executed.
            wait (`bool`, optional): If True, wait for the motion to be completed. Defaults to True.
            auto (`bool`, optional): If True, execute the paln without user confirmation. Defaults to False.

        Raises:
            ValueError: If the plan is not a `RobotTrajectory`.
        """
        assert isinstance(plan, RobotTrajectory)
        self.display_planned_path(plan)
        if auto is True:
            rospy.loginfo('Executing plan...')
            self.move_group.execute(plan, wait=wait)
            if wait is True:
                self.move_group.stop()
                rospy.loginfo("Finish execute")
        elif auto is False:
            self.confirm_to_execution()
            rospy.loginfo('Executing plan... {}'.format(wait))
            self.move_group.execute(plan, wait=wait)
            if wait is True:
                self.move_group.stop()
                rospy.loginfo("Finish execute")
        plan = None

    def plan_joint_move(self, joint_goal, start_state = None, wait = True):
        """ Plan a joint move

        Args:
            joint_goal (`list`): A list of joint angles in rad.
            wait (`bool`, optional): If True, wait for the motion to be completed. Defaults to True.
            auto (`bool`, optional): If True, execute the paln without user confirmation. Defaults to False.

        Raises:
            ValueError: If the plan is not a `RobotTrajectory`.
        """
        assert isinstance(joint_goal, list), 'joint_goal should be a list'
        if start_state is None:
            current_joint_state = self.move_group.get_current_joint_values()
        else:
            current_joint_state = start_state
        print(current_joint_state)
        joint_goal = [joint_goal[i] if joint_goal[i] is not None else current_joint_state[i] for i in range(len(joint_goal))]
        joint_goal_dict = dict(zip(self.move_group.get_joints(), joint_goal))
        print(joint_goal_dict)
        plan = self.move_group.plan(joints = joint_goal_dict)

        return plan


    def execute_joint_move(self, joint_goal, wait=True, auto=False):
        """Execute a joint move.

        Args:
            joint_goal (`list`): A list of joint angles in rad.
            wait (`bool`, optional): If True, wait for the motion to be completed. Defaults to True.
            auto (`bool`, optional): If True, execute the paln without user confirmation. Defaults to False.

        Raises:
            ValueError: If the plan is not a `RobotTrajectory`.
        """
        assert isinstance(joint_goal, list), 'joint_goal should be a list'

        current_joint_state = self.move_group.get_current_joint_values()
        joint_goal = [joint_goal[i] if joint_goal[i] is not None else current_joint_state[i] for i in range(len(joint_goal))]
        joint_goal_dict = dict(zip(self.move_group.get_joints(), joint_goal))
        print(self.move_group.get_joints())

        self.move_group.set_joint_value_target(joint_goal_dict)
        if auto is True:
            rospy.loginfo('Executing plan...')
            self.move_group.go(joints = True, wait=wait)
            if wait is True:
                self.move_group.stop()
                rospy.loginfo("Finish execute")
        elif auto is False:
            self.confirm_to_execution()
            rospy.loginfo('Executing plan... {}'.format(wait))
            self.move_group.go(joints = True, wait=wait)
            if wait is True:
                self.move_group.stop()
                rospy.loginfo("Finish execute")

        
    def is_plan_goal(self, plan, goal, eef_link=None, verbose=False):
        """Check if the end of the plan is the goal.

        Args:
            plan (`RobotTrajectory`): A motion plan.
            goal (`geometry_msgs/PoseStamped`): A pose target.
            eef_link (`str`, optional): End effector link. Defaults to None.
            verbose (`bool`, optional): If True, print the plan. Defaults to False.

        Returns:
            `bool`: If the end of the plan is the goal.
        """
        assert isinstance(plan, RobotTrajectory)
        assert isinstance(goal, PoseStamped)
        if len(plan.joint_trajectory.points) == 0:
            return False
        planned_goal_joint_states = plan.joint_trajectory.points[-1].positions
        planned_goal = self.get_forward_kinematics(planned_goal_joint_states)

        goal_vec = np.array([
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z,
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w
        ])
        planned_goal_vec = np.array([
            planned_goal.pose.position.x,
            planned_goal.pose.position.y,
            planned_goal.pose.position.z,
            planned_goal.pose.orientation.x,
            planned_goal.pose.orientation.y,
            planned_goal.pose.orientation.z,
            planned_goal.pose.orientation.w
        ])

        if not np.allclose(goal_vec, planned_goal_vec, atol=0.01):
            if verbose:
                rospy.loginfo('End of the planned path is not the goal.')
                rospy.logwarn('Planned goal: {}'.format(planned_goal_vec))
                rospy.logwarn('Goal: {}'.format(goal_vec))
            return False
        else:
            if verbose:
                rospy.loginfo('End of the planned path is the goal.')
            return True

    def is_goal_reached(self, plan, tolerance=0.01):
        """Check if the goal is reached.

        Args:
            tolerance (`float`, optional): Tolerance for checking if the goal is reached. Defaults to 0.01.

        Returns:
            `bool`: True if the goal is reached.
        """
        assert isinstance(plan, RobotTrajectory)
        goal_joint_state = plan.joint_trajectory.points[-1].positions
        current_joint_state = self.move_group.get_current_joint_values()
        return np.allclose(goal_joint_state, current_joint_state, atol=tolerance)

    def is_trajectory_continuous(self, plan, tolerance=0.1):
        """Check if the trajectory is continuous.

        Args:
            tolerance (`float`, optional): Tolerance for checking if the trajectory is continous. Defaults to 0.1.

        Returns:
            `bool`: True if the trajectory is continuous.
        """
        assert isinstance(plan, RobotTrajectory)
        for i in range(len(plan.joint_trajectory.points) - 1):
            if not np.allclose(plan.joint_trajectory.points[i].positions, plan.joint_trajectory.points[i+1].positions, atol=tolerance):
                return False
        return True

    def set_joint_constraints(self, joint_below_bound, joint_above_bound, joint_names):
        """Set joint constraints for motion

        Args:
            joint_below_bound (`float`): Joint value below the bound in Degrees.
            joint_above_bound (`float`): Joint value above the bound in Degrees.
            joint_names (`list` of `str`): Joint names.
        """
        if len(joint_names) != len(joint_below_bound) or len(joint_names) != len(joint_above_bound):
            raise ValueError('joint_names, joint_below_bound and joint_above_bound should have the same length.')

        joint_above_bound = np.deg2rad(joint_above_bound)
        joint_below_bound = np.deg2rad(joint_below_bound)
        joint_mean_bound = (joint_above_bound + joint_below_bound) / 2.0
        joint_above_tole = joint_above_bound - joint_mean_bound
        joint_below_tole = joint_mean_bound - joint_below_bound

        # get joint constraints
        constraints = Constraints()
        for i in range(len(joint_names)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_names[i]
            joint_constraint.position = joint_mean_bound[i]
            joint_constraint.tolerance_above = joint_above_tole[i]
            joint_constraint.tolerance_below = joint_below_tole[i]
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        self.move_group.set_path_constraints(constraints)
        rospy.logwarn('Set joint constraints: {}'.format(constraints))

    def get_path_constraints(self):
        """Get path constraints for motion

        Returns:
            `Constraints`: Joint constraints.
        """
        return self.move_group.get_path_constraints()

    def clear_joint_constraints(self):
        """Clear joint constraints."""
        self.move_group.clear_path_constraints()

    def confirm_to_execution(self):
        """Ask user to confirm to execute the plan."""
        while True:
            ask_for_execution = '[{}] Robot execution is requested. Excute? (y/n)'.format(self.group_name)
            rospy.logwarn(ask_for_execution)
            cmd = raw_input()

            if (cmd == 'y') or (cmd == 'Y'):
                rospy.logwarn('[{}] Got positive responce. Excute planned motion!'.format(self.group_name))
                break
            elif (cmd == 'n') or (cmd == 'N'):
                info_str = '[{}] Got ABORT signal. Shutdown!!'.format(self.group_name)
                rospy.logerr(info_str)
                raise Exception(info_str)
            else:
                rospy.logwarn('[{}] Command should be `y` or `n`. Got `{}`'.format(self.group_name, cmd))

    def scale_plan(self, plan, scale):
        assert isinstance(plan, RobotTrajectory)
        scale = float(scale)

        scaled_plan = copy.deepcopy(plan)
        # scale 
        for point in scaled_plan.joint_trajectory.points:
            time = point.time_from_start.to_sec()
            point.time_from_start = rospy.Duration(time * scale)

            point.velocities = [v/scale for v in point.velocities]
            point.accelerations = [a/scale for a in point.accelerations]
        return scaled_plan


    def set_max_velocity_scaling_factor(self, scale):
        self.move_group.set_max_velocity_scaling_factor(scale)


class PickingMotionPlanner(MotionPlanner):
    def __init__(self, group_name, pose_reference_frame):
        """Initialize PickingMotionPlanner.

        Args:
            group_name (`str`): MoveIt! group name.
            pose_reference_frame_id (`str`): Pose reference frame id.
        """
        super(PickingMotionPlanner, self).__init__(group_name, pose_reference_frame)

    def offset_pose_along_approach(self, pose, offset=-0.15):
        """Offset pose along approach direction.

        Args:
            pose (`PoseStamped`): A pose.
            offset (`float`, optional): Offset in [m]. Defaults to -0.15.

        Returns:
            `PoseStamped`: A pose with offset.
        """
        assert isinstance(pose, PoseStamped)
        assert isinstance(offset, float)

        # get target translation matrix
        pose_trans = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            ]
        pose_trans_mat = tf.transformations.translation_matrix(pose_trans)

        # get pose rotation matrix
        pose_quat = [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
            ]
        pose_rot_mat = tf.transformations.quaternion_matrix(pose_quat)

        # get z-axis (-) offset translation matrix
        approach_trans_mat = tf.transformations.translation_matrix([0., 0., offset])

        # get approach pose
        approach_trans_mat = np.linalg.multi_dot([pose_trans_mat, pose_rot_mat, approach_trans_mat])
        approach_pose = PoseStamped()
        approach_pose.header = pose.header
        approach_pose.pose.orientation = pose.pose.orientation
        approach_pose.pose.position.x = approach_trans_mat[0, 3]
        approach_pose.pose.position.y = approach_trans_mat[1, 3]
        approach_pose.pose.position.z = approach_trans_mat[2, 3]
        return approach_pose

    def set_start_state_to_current_state(self):
        """Set start robot state as current robot state
        """
        self.move_group.set_start_state_to_current_state()

    def set_start_state_from_plan(self, plan):
        """Set start robot state as last robot state of the plan.

        Args:
            plan (`RobotTrajectory`): A motion plan.
        """
        assert isinstance(plan, RobotTrajectory)
        last_joint_state = JointState()
        last_joint_state.header = plan.joint_trajectory.header
        last_joint_state.name = plan.joint_trajectory.joint_names
        last_joint_state.position = plan.joint_trajectory.points[-1].positions
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = last_joint_state
        self.move_group.set_start_state(moveit_robot_state)
        return last_joint_state

    def merge_plans(self, plans):
        """Merge multiple plans into one.

        Args:
            plans (`list` of `RobotTrajectory`): Plans to be merged.

        Returns:
            `RobotTrajectory`: A merged plan. If the plans are empty, return None.
        """
        # if plan is empty, return None
        if len(plans) == 0:
            rospy.logwarn('Plan list is empty.')
            return None

        # generate new empty plan
        merged_plan = RobotTrajectory()
        merged_plan.joint_trajectory.header = plans[0].joint_trajectory.header
        merged_plan.joint_trajectory.joint_names = plans[0].joint_trajectory.joint_names

        # append joint trajectory points
        last_time = rospy.Time()
        for plan in plans:
            # offset trajectory time
            for point in plan.joint_trajectory.points:
                point.time_from_start = rospy.Time(
                    last_time.secs + point.time_from_start.secs,
                    last_time.nsecs + point.time_from_start.nsecs)
            merged_plan.joint_trajectory.points += plan.joint_trajectory.points
            last_time = merged_plan.joint_trajectory.points[-1].time_from_start

        # remove duplicate points
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        return merged_plan

    def pose_to_angle(self, pose):
        """Convert a pose to joint angles.

        Args:
            pose (`PoseStamped`): A pose.

        Returns:
            `list` of `float`: Joint angles.
        """
        assert isinstance(pose, PoseStamped)
        target_pose = pose.pose

        trans = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z])
        target_angle = np.arctan2(trans[1], trans[0])

        return target_angle

    def remove_duplicate_points_on_plan(self, plan):
        """Remove duplicate points on a plan.

        Args:
            plan (`RobotTrajectory`): A plan.

        Returns:
            `RobotTrajectory`: A plan with no duplicate points.
        """
        point_num = len(plan.joint_trajectory.points)
        for i in range(point_num-1):
            try:
                if plan.joint_trajectory.points[i].positions == plan.joint_trajectory.points[i+1].positions:
                    plan.joint_trajectory.points.pop(i)
            except IndexError:
                pass
        return plan

    def bspline_motion_interpolation(self, start, goal, middle_poses, num_points):
        assert isinstance(goal, PoseStamped)
        assert isinstance(start, PoseStamped)

        print(middle_poses, "middle_poses")

        # get start and goal position
        start_pos = np.array([start.pose.position.x, start.pose.position.y, start.pose.position.z])
        goal_pos = np.array([goal.pose.position.x, goal.pose.position.y, goal.pose.position.z])

        # get start and goal orientation
        start_quat = np.array([start.pose.orientation.x, start.pose.orientation.y, start.pose.orientation.z, start.pose.orientation.w])
        goal_quat = np.array([goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])

        # interpolate quaternion
        quat_list = []
        for i in range(num_points):
            quat_list.append(tf.transformations.quaternion_slerp(start_quat, goal_quat, i / (num_points - 1.0)))

        # get watypoints
        xyz = []
        xyz.append(start_pos)
        for p in middle_poses:
            xyz.append(p)
        xyz.append(goal_pos)
        xyz = np.array(xyz)
        xyz = np.transpose(xyz)

        # set weight
        weight = np.ones(xyz.shape[1])
        weight[0] = 10
        weight[-1] = 10

        # interpolate the pose
        print("----------------")
        print(xyz, "xyz")
        print("----------------")
        tck, u = splprep(xyz, w=weight, s=0.05, k=2)
        u_new = np.linspace(0, 1, num_points, endpoint=True)
        x_new, y_new, z_new = splev(u_new, tck)

        # make pose
        pose_list = []
        for i in range(num_points):
            pose = Pose()
            pose.position.x = x_new[i]
            pose.position.y = y_new[i]
            pose.position.z = z_new[i]
            pose.orientation.x = quat_list[i][0]
            pose.orientation.y = quat_list[i][1]
            pose.orientation.z = quat_list[i][2]
            pose.orientation.w = quat_list[i][3]
            pose_list.append(pose)
        return pose_list


class CupMotionPlanner(PickingMotionPlanner):
    def __init__(self, group_name, pose_reference_frame_id):
        """Initialize CupMotionPlanner.

        Args:
            group_name (`str`): MoveIt! group name.
            pose_reference_frame_id (`str`): Pose reference frame id.
        """
        super(CupMotionPlanner, self).__init__(group_name, pose_reference_frame_id)

    def get_pick_plan(self, cup_pose, standby_joint_config, standby_pose, z_offset=0.1):
        assert isinstance(cup_pose, PoseStamped)

        plans = []
        # 1)  assert cur pose is standby pose
        joint_state = np.array(self.move_group.get_current_joint_values())
        
        if not np.allclose(joint_state, np.array(standby_joint_config), atol=0.01):
            cup_stb_plan = self.plan_joint_move(standby_joint_config)
            plans.append(cup_stb_plan)

        # 2) approach pose plan
        # 2.1 approach_start
        approach_start = standby_pose

        # 2.2 approach pose
        approach_pose = copy.deepcopy(cup_pose)
        approach_pose.pose.position.z += z_offset

        # 2.3 cup pose
        if len(plans) > 0:
            self.set_start_state_from_plan(plans[0])

        # get plan
        plan, fraction = self.move_group.compute_cartesian_path(
            [approach_pose.pose, cup_pose.pose],
            0.015,
            0.0,
        )
        if fraction != 1.0:
            raise ValueError('Cup Pick plan fails. Fraction is not 1.0 bur {}'.format(fraction))

        plans.append(plan)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan

    def get_lift_up_plan(self, zero_pose, cup_place_pose, z_offset):
        plans = []

        # 1) Go to zero pose
        # 1.1) get curpose
        current_pose = self.get_current_pose()

        # 1.2) lift up pose
        lift_up_pose = copy.deepcopy(current_pose)
        lift_up_pose.pose.position.z += z_offset

        # 1.3) x_0
        x_0 = copy.deepcopy(lift_up_pose)
        x_0.pose.position.x = zero_pose[0]
        x_0.pose.position.y = zero_pose[1]
        x_0.pose.position.z = zero_pose[2]

        orientation_mtx = np.array([[0, 0, 1, 0],
                                   [-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 0, 1]])
        final_orientation = tf.transformations.quaternion_from_matrix(orientation_mtx)
        x_0.pose.orientation.x = final_orientation[0]
        x_0.pose.orientation.y = final_orientation[1]
        x_0.pose.orientation.z = final_orientation[2]
        x_0.pose.orientation.w = final_orientation[3]

        # 1.4) get plan
        self.set_start_state_to_current_state()
        waypoints = []
    
        waypoints.append(lift_up_pose.pose)
        waypoints.append(x_0.pose)

        plan_zero, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )
        
        if fraction != 1.0:
            raise ValueError('Cup Pick plan failes. Fraction is not 1.0 but {}'.format(fraction))

        plans.append(plan_zero)

        # 2) Drop off approach

        drop_angle = self.pose_to_angle(cup_place_pose)
        last_joint_state = self.set_start_state_from_plan(plans[0])
        drop_plan = self.plan_joint_move(
            [drop_angle, None, None, None, None, None], start_state=last_joint_state.position)
        plans.append(drop_plan)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan



    def get_place_plan(self, place_pose, z_offset):
        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) place_pose
        place_pose.pose.orientation.x = current_pose.pose.orientation.x
        place_pose.pose.orientation.y = current_pose.pose.orientation.y
        place_pose.pose.orientation.z = current_pose.pose.orientation.z
        place_pose.pose.orientation.w = current_pose.pose.orientation.w

        waypoints = [place_pose.pose]
        self.move_group.set_start_state_to_current_state()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError(
                'Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        return plan

    def get_standby_plan(self, back_pose, z_offset):
        plans = []

        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) Lift Up
        lift_up_pose = copy.deepcopy(current_pose)
        lift_up_pose.pose.position.z += z_offset
    
        # 3) Middle
        middle_pose = copy.deepcopy(back_pose)
        middle_pose.pose.position.z = lift_up_pose.pose.position.z

        # 3) Go Back
        back_pose.pose.position.z = middle_pose.pose.position.z

        waypoints = []
        waypoints.append(lift_up_pose.pose)
        waypoints.append(back_pose.pose)

        self.move_group.set_start_state_to_current_state()
        plan_away, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError('Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))

        plans.append(plan_away)

        # To back to 0 pose
        last_joint_state = self.set_start_state_from_plan(plans[0])
        plan_zero = self.plan_joint_move(
            [0, None, None, None, None, None], start_state=last_joint_state.position)
        plans.append(plan_zero)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan

    def get_standby_plan_old(self, standby_pose, middle_pose, z_offset):
        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) go back pose
        back_pose = self.offset_pose_along_approach(current_pose, -0.03)

        # 2) lift up pose
        lift_up_pose = copy.deepcopy(back_pose)
        lift_up_pose.pose.position.z += z_offset

        # 3) place approach pose
        approach_pose = self.offset_pose_along_approach(lift_up_pose, -0.15)
        approach_pose.pose.position.z += 0.05

        # 4) get bspline poses
        bspline_poses = self.bspline_motion_interpolation(
            start=approach_pose,
            goal=standby_pose,
            middle_poses=[middle_pose],
            num_points=50,
        )

        # 5) get plan
        waypoints = []
        waypoints.append(back_pose.pose)
        waypoints.append(lift_up_pose.pose)
        for p in bspline_poses:
            waypoints.append(p)

        if True:
            p = PoseArray()
            p.header.frame_id = self.pose_reference_frame
            p.header.stamp = rospy.Time.now()
            p.poses = waypoints
            for i in range(5):
                self.debug_pose_array_pub.publish(p)
                rospy.sleep(0.1)

        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )
        if fraction != 1.0:
            raise ValueError('Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        return plan


class PlateMotionPlanner(PickingMotionPlanner):
    def __init__(self, group_name, pose_reference_frame_id):
        """Initialize PlateMotionPlanner.

        Args:
            group_name (`str`): MoveIt! group name.
            pose_reference_frame_id (`str`): Pose reference frame id.
        """
        super(PlateMotionPlanner, self).__init__(group_name, pose_reference_frame_id)

    def get_pickup_plan(self, plate_pose, place_app_joint_cfg, standby_joint_config, standby_pose, plate_stb_pose, z_offset=0.1):
        plans = []
        assert isinstance(plate_pose, PoseStamped)
        cur_pose = self.get_current_pose()

        # 1) assert cur pose is standby pose
        joint_state = np.array(self.move_group.get_current_joint_values())
        if not np.allclose(joint_state, np.array(standby_joint_config), atol=0.01):
            plan = self.plan_joint_move(standby_joint_config)
            plan = self.remove_duplicate_points_on_plan(plan)
            plans.append(plan)


        # 2) approach pose
        approach_pose_1 = self.offset_pose_along_approach(plate_pose, z_offset)
        approach_pose_1.pose.position.z += 0.05
        approach_pose_1.pose.position.y -= 0.05
        if len(plans) > 0:
            self.set_start_state_from_plan(plans[-1])
            last_joint_state = self.set_start_state_from_plan(plans[-1]).position
        else:
            self.set_start_state_to_current_state()
            last_joint_state = self.move_group.get_current_joint_values()
        
        #plan = self.get_plan(approach_pose_1)
        plan = self.plan_joint_move(place_app_joint_cfg, start_state = last_joint_state)
        plans.append(plan)

        # 3) Pick UP
        approach_pose_2 = self.offset_pose_along_approach(plate_pose, z_offset)
        bspline_poses = self.bspline_motion_interpolation(
            start = approach_pose_1,
            goal = plate_pose,
            middle_poses=[[approach_pose_2.pose.position.x, approach_pose_2.pose.position.y, approach_pose_2.pose.position.z]],
            num_points=40,
        )
        waypoints = []
        for p in bspline_poses:
            waypoints.append(p)
        if True:
            p = PoseArray()
            p.header.frame_id = self.pose_reference_frame
            p.header.stamp = rospy.Time.now()
            p.poses = waypoints
            for i in range(5):
                self.debug_pose_array_pub.publish(p)
                rospy.sleep(0.1)
        # get plan
        self.set_start_state_from_plan(plans[-1])
        plan_2, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.015,
            0.0,
        )
        if fraction != 1.0:
            raise ValueError(
                'Plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        plans.append(plan_2)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan

    def get_lift_up_plan(self, place_pose, middle_pose, zero_pose, z_offset=0.1):
        plans = []
        # 1.1) get curpose
        current_pose = self.get_current_pose()
        
        # 1.2) lift up pose
        lift_up_pose = copy.deepcopy(current_pose)
        lift_up_pose.pose.position.z += z_offset

        # 1.3) plan
        self.set_start_state_to_current_state()
        waypoints = [lift_up_pose.pose]
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )
        plans.append(plan)
        

        # 2.1) plate 0 pose
        z_rotate_90_mtx = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        orientation_now = np.array([current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w])
        orientation_now_mtx = tf.transformations.quaternion_matrix(orientation_now)
        orientation_rotate_90 = np.dot(z_rotate_90_mtx, orientation_now_mtx)

        x_0 = copy.deepcopy(lift_up_pose)
        x_0.pose.position.x = zero_pose[0]
        x_0.pose.position.y = zero_pose[1]
        x_0.pose.position.z = zero_pose[2]

        final_orientation = tf.transformations.quaternion_from_matrix(
            orientation_rotate_90)
        x_0.pose.orientation.x = final_orientation[0]
        x_0.pose.orientation.y = final_orientation[1]
        x_0.pose.orientation.z = final_orientation[2]
        x_0.pose.orientation.w = final_orientation[3]

        # 2.2) get bspline poses
        bspline_poses = self.bspline_motion_interpolation(
            start=lift_up_pose,
            goal=x_0,
            middle_poses=[middle_pose],
            num_points=50,
        )

        # 2.3) get plan
        waypoints = []
        for p in bspline_poses:
            waypoints.append(p)

        if True:
            p = PoseArray()
            p.header.frame_id = self.pose_reference_frame
            p.header.stamp = rospy.Time.now()
            p.poses = waypoints
            for i in range(5):
                self.debug_pose_array_pub.publish(p)
                rospy.sleep(0.1)

        self.set_start_state_from_plan(plans[-1])
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError(
                'Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        plans.append(plan)

        # 2) approach pose
        last_joint_state = self.set_start_state_from_plan(plans[-1])
        angle_j0 = self.pose_to_angle(place_pose)
        plan = self.plan_joint_move(
            [angle_j0, None, None, None, None, None], start_state=last_joint_state.position)
        plans.append(plan)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan

    def get_place_plan(self, place_pose, z_offset=0.1):
        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) place_pose
        place_pose.pose.orientation.x = current_pose.pose.orientation.x
        place_pose.pose.orientation.y = current_pose.pose.orientation.y
        place_pose.pose.orientation.z = current_pose.pose.orientation.z
        place_pose.pose.orientation.w = current_pose.pose.orientation.w

        waypoints = [place_pose.pose]
        self.move_group.set_start_state_to_current_state()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError(
                'Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        return plan

    def get_place_plan_old(self, place_pose, middle_pose, z_offset):
        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) lift up pose
        lift_up_pose = copy.deepcopy(current_pose)
        lift_up_pose.pose.position.z += z_offset

        # 3) place approach pose
        approach_pose = self.offset_pose_along_approach(place_pose, -0.15)
        approach_pose.pose.position.z += 0.05

        # 4) get bspline poses
        bspline_poses = self.bspline_motion_interpolation(
            start=lift_up_pose,
            goal=approach_pose,
            middle_poses=[middle_pose],
            num_points=50,
        )

        # 5) get plan
        waypoints = []
        for p in bspline_poses:
            waypoints.append(p)
        waypoints.append(place_pose.pose)

        if True:
            p = PoseArray()
            p.header.frame_id = self.pose_reference_frame
            p.header.stamp = rospy.Time.now()
            p.poses = waypoints
            for i in range(5):
                self.debug_pose_array_pub.publish(p)
                rospy.sleep(0.1)

        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError('Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        return plan

    def get_standby_plan(self, back_pose, plate_0_pose):
        plans = []
        # 1) get curpose
        current_pose = self.get_current_pose()

        # 2) go back pose
        waypoints = [back_pose.pose]
        self.move_group.set_start_state_to_current_state()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            0.03,
            0.0
        )

        if fraction != 1.0:
            raise ValueError(
                'Cup Pick plan failes. Fraction is not 1.0 bur {}'.format(fraction))
        plans.append(plan)

        # 3) go plate_0_pose
        angle_0_pose = self.pose_to_angle(plate_0_pose)
        last_joint_config = self.set_start_state_from_plan(plans[-1]).position
        plan = self.plan_joint_move(
            [angle_0_pose, None, None, None, None, None], start_state=last_joint_config)
        plans.append(plan)

        # Reset _end_effector_link to zero pose
        self.set_start_state_from_plan(plans[-1])
        self.move_group.set_pose_target(plate_0_pose.pose)
        plan = self.move_group.plan()
        plans.append(plan)

        merged_plan = self.merge_plans(plans)
        merged_plan = self.remove_duplicate_points_on_plan(merged_plan)
        self.display_planned_path(merged_plan)

        return merged_plan