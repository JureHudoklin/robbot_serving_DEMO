#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped


class Indy7CalibrationDataManager(object):
    """Calibration data manager for the indy7_handeye package."""
    def __init__(self, config_dir):
        """Initialize the calibration data manager.

        Args:
            config_dir (str): The directory of the configuration files.
        """
        # define the directory where the calibration data is stored
        self.config_dir = config_dir
        self.data_dir = os.path.join(config_dir, 'data')
        self.opencv_dir = os.path.join(self.data_dir, 'opencv')
        self.zivid_dir = os.path.join(self.data_dir, 'zivid')
        self.pose_history_dir = os.path.join(self.data_dir, 'pose_history')

        # create the directories if they don't exist
        if not os.path.exists(self.zivid_dir):
            rospy.logwarn('Zivid data directory does not exist. Creating {}'.format(self.zivid_dir))
            os.makedirs(self.zivid_dir)
        if not os.path.exists(self.opencv_dir):
            rospy.logwarn('OpenCV data directory does not exist. Creating {}'.format(self.opencv_dir))
            os.makedirs(self.opencv_dir)
        if not os.path.exists(self.pose_history_dir):
            rospy.logwarn('Pose history directory does not exist. Creating {}'.format(self.pose_history_dir))
            os.makedirs(self.pose_history_dir)

        # create the file prefixes
        self.raw_img_prefix = 'raw_img{:02d}.png'
        self.chess_img_prefix = 'chess_img{:02d}.png'
        self.zdf_prefix = 'img{:02d}.zdf'
        self.pos_prefix = 'pos{:02d}.yaml'

    def save_pose_history(self, idx, pose):
        """Save the pose history.

        Args:
            idx (int): The index of the data point to save. Range: [1, 100)
            pose (PoseStamped): Current pose of the gripper.
        """
        self.assert_idx_in_range(idx)

        translation = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        quaternion = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        pose_history_file = os.path.join(self.pose_history_dir, self.pos_prefix.format(idx))
        fs = cv2.FileStorage(pose_history_file, cv2.FILE_STORAGE_WRITE)
        fs.write(name='frame_id', val=pose.header.frame_id)
        fs.write(name='position', val=translation)
        fs.write(name='orientation', val=quaternion)
        fs.release()
        rospy.loginfo('Saved pose history {}'.format(idx))

    def load_pose_history(self, idx):
        """Load the pose history.

        Args:
            idx (int): The index of the data point to load. Range: [1, 100)

        Returns:
            PoseStamped: The pose of the gripper at the given index. If the file does not exist, None is returned.
        """
        self.assert_idx_in_range(idx)
        pose_history_file = os.path.join(self.pose_history_dir, self.pos_prefix.format(idx))
        if os.path.exists(pose_history_file):
            fs = cv2.FileStorage(pose_history_file, cv2.FILE_STORAGE_READ)
            frame_id = fs.getNode('frame_id').string()
            position = fs.getNode('position').mat().squeeze()
            orientation = fs.getNode('orientation').mat().squeeze()
            fs.release()

            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = position[0]
            pose.pose.position.y = position[1]
            pose.pose.position.z = position[2]
            pose.pose.orientation.x = orientation[0]
            pose.pose.orientation.y = orientation[1]
            pose.pose.orientation.z = orientation[2]
            pose.pose.orientation.w = orientation[3]
            return pose
        else:
            rospy.logwarn('Pose history {:02d} does not exist'.format(idx))
            return None

    def save_opencv_data(self, idx, raw_img, chess_img, tf_gripper2base, tf_target2cam):
        """Save the opencv data.

        Args:
            idx (int): The index of the data point to save. Range: [1, 100)
            raw_img (numpy.ndarray): The raw image with BGR.
            chess_img (numpy.ndarray): The chessboard image with BGR.
            tf_gripper2base (numpy.ndarray): The transformation from the gripper to the base.
            tf_target2cam (numpy.ndarray): The transformation from the target to the camera.
        """
        self.assert_idx_in_range(idx)

        # save images for debuging
        raw_img_file = os.path.join(self.opencv_dir, self.raw_img_prefix.format(idx))
        cv2.imwrite(raw_img_file, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
        rospy.loginfo('Saved raw image {}'.format(raw_img_file))
        chess_img_file = os.path.join(self.opencv_dir, self.chess_img_prefix.format(idx))
        cv2.imwrite(chess_img_file, cv2.cvtColor(chess_img, cv2.COLOR_RGB2BGR))
        rospy.loginfo('Saved chess image {}'.format(chess_img_file))

        # save the transformation
        opencv_pos_file = os.path.join(self.opencv_dir, self.pos_prefix.format(idx))
        fs = cv2.FileStorage(opencv_pos_file, cv2.FILE_STORAGE_WRITE)
        fs.write(name='tf_gripper2base', val=tf_gripper2base)
        fs.write(name='tf_target2cam', val=tf_target2cam)
        fs.release()
        rospy.loginfo('Saving data point(gripper2base, target2cam) to file: {}'.format(opencv_pos_file))

    def load_opencv_data(self, idx):
        """Load the opencv data.

        Args:
            idx (int): The index of the data point to load. Range: [1, 100)

        Returns:
            (numpy.ndarray, numpy.ndarray): The transformation from the gripper to the base and the target to the camera. If the file does not exist, None is returned.
        """
        self.assert_idx_in_range(idx)
        opencv_pos_file = os.path.join(self.opencv_dir, self.pos_prefix.format(idx))
        if os.path.exists(opencv_pos_file):
            fs = cv2.FileStorage(opencv_pos_file, cv2.FILE_STORAGE_READ)
            tf_gripper2base = fs.getNode('tf_gripper2base').mat()
            tf_target2cam = fs.getNode('tf_target2cam').mat()
            fs.release()
            return tf_gripper2base, tf_target2cam
        else:
            rospy.logwarn('OpenCV data point {:02d} does not exist'.format(idx))
            return None, None

    def save_zivid_data(self, idx, tf_gripper2base):
        """Save the Zivid data.

        Capture the scene and save it with Zivid python3 API. Zivid ROS node
        must be stopped before executing this function. If the Zivid node is
        running, it will try to kill it. The gripper to base transformation is
        saved with (mm) unit.

        Args:
            idx (`int`): The index of the data point to save. Range: [1, 100)
            tf_gripper2base (`numpy.ndarray`): The transformation from gripper to base.
        """
        self.assert_idx_in_range(idx)

        # convert tf to mm scale for zivid handeye calibration CLI
        tf_gripper2base_mm = np.copy(tf_gripper2base)
        tf_gripper2base_mm[:3, 3] *= 1000.0

        # save the transformation
        zivid_pos_file = os.path.join(self.zivid_dir, self.pos_prefix.format(idx))
        fs = cv2.FileStorage(zivid_pos_file, cv2.FILE_STORAGE_WRITE)
        fs.write(name='PoseState', val=tf_gripper2base_mm)
        fs.release()
        rospy.loginfo('Saving data point(gripper2base) to file: {}'.format(zivid_pos_file))

        # save zivid point cloud data
        zdf_img_file = os.path.join(self.zivid_dir, self.zdf_prefix.format(idx))
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        ret = os.system('python3 {} --file {}'.format(os.path.join(this_file_dir, 'calibration_capture_zivid_api.py'), zdf_img_file))
        if ret != 0:
            rospy.logwarn('Failed to capture zivid point cloud data. Kill node and try again.')
            os.system('rosnode kill /zivid_camera/zivid_camera')
            rospy.sleep(2.0)
            os.system('python3 {} --file {}'.format(os.path.join(this_file_dir, 'calibration_capture_zivid_api.py'), zdf_img_file))
        rospy.loginfo('Saved zivid point cloud data to file: {}'.format(zdf_img_file))

    def is_data_valid(self, idx, verbose=False):
        """Check the data point is valid.

        Args:
            idx (int): The index of the data point to check. Range: [1, 100)
            verbose (bool, optional): If True, print the error message. Defaults to False.

        Returns:
            bool: True if the data point is valid, False otherwise.
        """
        self.assert_idx_in_range(idx)

        # get file names
        raw_img_file = os.path.join(self.opencv_dir, self.raw_img_prefix.format(idx))
        chess_img_file = os.path.join(self.opencv_dir, self.chess_img_prefix.format(idx))
        opencv_pos_file = os.path.join(self.opencv_dir, self.pos_prefix.format(idx))
        zivid_img_file = os.path.join(self.zivid_dir, self.zdf_prefix.format(idx))
        zivid_pos_file = os.path.join(self.zivid_dir, self.pos_prefix.format(idx))
        pose_history_file = os.path.join(self.pose_history_dir, self.pos_prefix.format(idx))

        # check if the files exist
        raw_img_exist = os.path.exists(raw_img_file)
        chess_img_exist = os.path.exists(chess_img_file)
        opencv_pos_exist = os.path.exists(opencv_pos_file)
        zivid_img_exist = os.path.exists(zivid_img_file)
        zivid_pos_exist = os.path.exists(zivid_pos_file)
        pose_history_exist = os.path.exists(pose_history_file)

        # return the result
        if not (raw_img_exist and chess_img_exist and opencv_pos_exist and zivid_img_exist and zivid_pos_exist and pose_history_exist):
            if not raw_img_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(raw_img_file))
            if not chess_img_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(chess_img_file))
            if not opencv_pos_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(opencv_pos_file))
            if not zivid_img_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(zivid_img_file))
            if not zivid_pos_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(zivid_pos_file))
            if not pose_history_exist and verbose:
                rospy.logwarn('Can\'t find {}'.format(pose_history_file))
            return False
        rospy.loginfo('Data point {:02d} is valid'.format(idx))
        return True

    def assert_idx_in_range(self, idx):
        """Assert the index is in the range.

        Args:
            idx (int): The index of the data point to check. Range: [1, 100)
        """
        assert 1 <= idx <= 100, 'Index out of range: {}'.format(idx)

    def delete_data(self, idx, delete_pose_history=False, verbose=False):
        """Delete the data point.

        Args:
            idx (int): The index of the data point to delete. Range: [1, 100)
        """
        self.assert_idx_in_range(idx)

        # get file names
        raw_img_file = os.path.join(self.opencv_dir, self.raw_img_prefix.format(idx))
        chess_img_file = os.path.join(self.opencv_dir, self.chess_img_prefix.format(idx))
        opencv_pos_file = os.path.join(self.opencv_dir, self.pos_prefix.format(idx))
        zivid_img_file = os.path.join(self.zivid_dir, self.zdf_prefix.format(idx))
        zivid_pos_file = os.path.join(self.zivid_dir, self.pos_prefix.format(idx))
        pose_history_file = os.path.join(self.pose_history_dir, self.pos_prefix.format(idx))

        # delete the files
        if os.path.exists(raw_img_file):
            os.remove(raw_img_file)
            if verbose:
                rospy.loginfo('Deleted {}'.format(raw_img_file))
        if os.path.exists(chess_img_file):
            os.remove(chess_img_file)
            if verbose:
                rospy.logwarn('Deleted {}'.format(chess_img_file))
        if os.path.exists(opencv_pos_file):
            os.remove(opencv_pos_file)
            if verbose:
                rospy.logwarn('Deleted {}'.format(opencv_pos_file))
        if os.path.exists(zivid_img_file):
            os.remove(zivid_img_file)
            if verbose:
                rospy.logwarn('Deleted {}'.format(zivid_img_file))
        if os.path.exists(zivid_pos_file):
            os.remove(zivid_pos_file)
            if verbose:
                rospy.logwarn('Deleted {}'.format(zivid_pos_file))
        if delete_pose_history and os.path.exists(pose_history_file):
            os.remove(pose_history_file)
            rospy.logwarn('Deleted {}'.format(pose_history_file))
        rospy.logwarn('Deleted data point: {}'.format(idx))

    def sort_data(self, verbose=False):
        """Sort the data points with continous sequential idx.

        If the data point is invalid, delete it and sort the index.

        Args:
            verbose (bool, optional): If True, print the error message. Defaults to False.
        """
        # delete invalid data points
        valid_idx = []
        for i in range(1, 100):
            if not self.is_data_valid(i, verbose):
                self.delete_data(i, verbose=verbose)
            else:
                valid_idx.append(i)
        valid_idx.sort()

        # rename the files to continuous index
        for new_index, old_index in enumerate(valid_idx, 1):
            if new_index != old_index:
                os.rename(
                    os.path.join(self.opencv_dir, self.raw_img_prefix.format(old_index)),
                    os.path.join(self.opencv_dir, self.raw_img_prefix.format(new_index)))
                os.rename(
                    os.path.join(self.opencv_dir, self.chess_img_prefix.format(old_index)),
                    os.path.join(self.opencv_dir, self.chess_img_prefix.format(new_index)))
                os.rename(
                    os.path.join(self.opencv_dir, self.pos_prefix.format(old_index)),
                    os.path.join(self.opencv_dir, self.pos_prefix.format(new_index)))
                os.rename(
                    os.path.join(self.zivid_dir, self.zdf_prefix.format(old_index)),
                    os.path.join(self.zivid_dir, self.zdf_prefix.format(new_index)))
                os.rename(
                    os.path.join(self.zivid_dir, self.pos_prefix.format(old_index)),
                    os.path.join(self.zivid_dir, self.pos_prefix.format(new_index)))
                os.rename(
                    os.path.join(self.pose_history_dir, self.pos_prefix.format(old_index)),
                    os.path.join(self.pose_history_dir, self.pos_prefix.format(new_index)))

    def save_handeye_data(self, tf_cam2gripper, method):
        """Save the handeye calibration data.

        Args:
            tf_cam2gripper (`numpy.ndarray`): The transformation matrix from camera to gripper.
            method (`str`): The method used to calculate the transformation matrix.
        """
        handeye_file = os.path.join(self.config_dir, 'cam2gripper.yaml')
        fs = cv2.FileStorage(handeye_file, cv2.FILE_STORAGE_WRITE)
        fs.write('tf_cam2gripper', tf_cam2gripper)
        fs.write('method', method)
        fs.release()
        rospy.loginfo('Saved handeye data to {}'.format(handeye_file))

    def load_handeye_data(self):
        """Load the handeye calibration data.

        Returns:
            `numpy.ndarray`: The transformation matrix from camera to gripper.
            `str`: The method used to calculate the transformation matrix.
        """
        handeye_file = os.path.join(self.config_dir, 'cam2gripper.yaml')
        fs = cv2.FileStorage(handeye_file, cv2.FILE_STORAGE_READ)
        tf_cam2gripper = fs.getNode('tf_cam2gripper').mat()
        method = fs.getNode('method').string()
        fs.release()
        rospy.loginfo('Loaded handeye data from {}'.format(handeye_file))
        return tf_cam2gripper, method

    @property
    def pose_history_indices(self):
        """Get the indices of the pose history.

        Returns:
            list: The indices of the pose history.
        """
        file_list = []
        for i in range(1, 100):
            file = os.path.join(self.pose_history_dir, self.pos_prefix.format(i))
            if os.path.exists(file):
                file_list.append(file)
        file_list.sort()
        return [int(filter(str.isdigit, os.path.basename(file_name))) for file_name in file_list]
