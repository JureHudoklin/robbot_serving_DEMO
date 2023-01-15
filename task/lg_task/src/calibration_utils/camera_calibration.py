#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

import tf
import tf.transformations


class CameraCalibration(object):
    """Camera calibration"""
    def __init__(self, start_corner, end_corner, square_size):
        """Initialize the calibration data manager.

        Args:
            start_corner(`list`): The start corner of the chessboard [x, y] (mm).
            end_corner(`list`): The end corner of the chessboard [x, y] (mm).
            square_size(`float`): The size of the chessboard square in (mm).

        """
        self.start_corner = np.array(start_corner)
        self.end_corner = np.array(end_corner)
        self.square_size = float(square_size)
        self.pattern_size = np.abs((self.end_corner - self.start_corner)/square_size)
        if np.sum(self.pattern_size % 1) > 0:
            e = 'The pattern size is not a multiple of the square size. {}'
            raise ValueError(e.format(self.pattern_size))
        else:
            self.pattern_size = np.array(self.pattern_size, dtype=int) + 1
            self.pattern_size = tuple(self.pattern_size)

    def get_obj_points(self, chessboard_pose=None):
        """Get chessboard object points.

        Args:
            chessboard_pose (`numpy.ndarray`, optional): Chessboard pose transformation. Defaults to None.

        Returns:
            `numpy.ndarray`: (N, 3) object points.
        """
        x = np.linspace(
            self.start_corner[0],
            self.end_corner[0],
            self.pattern_size[0],
            endpoint=True)/1000.0
        y = np.linspace(
            self.start_corner[1],
            self.end_corner[1],
            self.pattern_size[1],
            endpoint=True)/1000.0
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        obj_points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        if chessboard_pose is not None:
            obj_points_homogeneous = np.hstack((obj_points, np.ones((obj_points.shape[0], 1))))
            obj_points_homogeneous = np.dot(chessboard_pose, obj_points_homogeneous.T).T
            obj_points = obj_points_homogeneous[:, 0:3]
        return obj_points

    def get_img_points(self, img, is_border_white=False, vis=False):
        """Get chessboard image points from chessboard image.

        Args:
            img ('numpy.ndarray'): BGR image.
            is_border_white (`bool`, optional): If true, the border of the chessboard is white. Defaults to False.
            vis (`bool`, optional): If True, show chessboard image.
            Defaults to False.

        Returns:
            `numpy.ndarray`: (N, 2) image points.
            `numpy.ndarray`: Chessboard corner image with BGR.
        """
        # get gray image from the image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not is_border_white:
            gray_img = cv2.bitwise_not(gray_img)

        # find chessboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(
            gray_img,
            self.pattern_size,
            flags=flags)
        if ret is True:
            # criteria for refining corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1),
                                       criteria)
        else:
            return None, None

        # Draw and display the corners
        cv2.drawChessboardCorners(img, self.pattern_size, corners, ret)
        if vis is True:
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return corners, img

    def get_target2cam(self, img, camera_intr, is_border_white=False, chessboard_pose=None, vis=False):
        """Get target to camera transformation.

        Args:
            img (`numpy.ndarray`): BGR image.
            camera_intr (`numpy.ndarray`): (3, 3) camera intrinsic matrix.
            is_border_white (`bool`, optional): If true, the border of the chessboard is white. Defaults to False.
            chessboard_pose (`numpy.ndarray`, optional): (4, 4) chessboard pose. Defaults to None.
            vis (`bool`, optional): If ture, visualize chessborad corner image. Defaults to False.
        Returns:
            `numpy.ndarray`: (4, 4) target to camera transformation.
            `numpy.ndarray`: Chessboard corner image with BGR.
        """
        # get object points and image points
        obj_points = self.get_obj_points(chessboard_pose)
        img_points, corner_img = self.get_img_points(img, is_border_white, vis=vis)

        # set initial paramters
        dist_coeffs = np.zeros((4, 1))
        flags = cv2.SOLVEPNP_ITERATIVE

        # solve pnp
        (retval, r_vector, t_vector) = cv2.solvePnP(obj_points,
                                                    img_points,
                                                    camera_intr,
                                                    dist_coeffs,
                                                    flags=flags)
        r_matrix = np.zeros((3, 3))
        cv2.Rodrigues(r_vector, r_matrix)

        # get transform
        tf_target2cam = np.eye(4)
        tf_target2cam[0:3, 0:3] = r_matrix
        tf_target2cam[0:3, 3] = t_vector.squeeze()
        return tf_target2cam, corner_img

    def inverse(self, tf_matrix):
        """Get camera to target transformation.

        Args:
            tf_matrix (`numpy.ndarray`): (4, 4) target to camera transformation.

        Returns:
            `numpy.ndarray`: (4, 4) camera to target transformation.
        """
        r = tf_matrix[0:3, 0:3].T
        t = -np.dot(r, tf_matrix[0:3, 3])
        tf_matrix_inv = np.eye(4)
        tf_matrix_inv[0:3, 0:3] = r
        tf_matrix_inv[0:3, 3] = t
        return tf_matrix_inv

    def get_gripper2base(self, pose_stamped):
        """Get gripper to base transformation.

        Args:
            pose_stamped (`PoseStamped`): PoseStamped message.

        Returns:
            `numpy.ndarray`: (4, 4) gripper to base transformation.
        """        
        # make translation and quaternion
        translation = np.array([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z])
        quaternion = np.array([
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w])
        tf_gripper2base = tf.transformations.quaternion_matrix(quaternion)
        tf_gripper2base[0:3, 3] = translation
        return tf_gripper2base

    def get_cam2gripper(self, tf_target2cam_list, tf_gripper2base_list, method):
        """Get camera to gripper transformation.

        Args:
            tf_target2cam_list (A `list` of `numpy.ndarray`): (4, 4) target to camera transformation.
            tf_gripper2base_list (A `list` of `numpy.ndarray`): (4, 4) gripper to base transformation.

        Returns:
            `numpy.ndarray`: (4, 4) camera to gripper transformation.
        """
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        for tf_target2cam, tf_gripper2base in zip(tf_target2cam_list, tf_gripper2base_list):
            R_gripper2base_list.append(tf_gripper2base[0:3, 0:3])
            t_gripper2base_list.append(tf_gripper2base[0:3, 3:4]*1000.0)
            R_target2cam_list.append(tf_target2cam[0:3, 0:3])
            t_target2cam_list.append(tf_target2cam[0:3, 3:4]*1000.0)

        # get camera to gripper transformation
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base_list,
            t_gripper2base=t_gripper2base_list,
            R_target2cam=R_target2cam_list,
            t_target2cam=t_target2cam_list,
            method=method)

        tf_cam2gripper = np.eye(4)
        tf_cam2gripper[0:3, 0:3] = R_cam2gripper
        tf_cam2gripper[0:3, 3] = t_cam2gripper.squeeze()*0.001
        return tf_cam2gripper
