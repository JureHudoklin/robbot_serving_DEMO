#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import numpy as np
import scipy.signal as ssg

import rospy
import ros_numpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


class VisionSensorInterface(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.Subscriber('/camera/color/image_raw', Image, self.read_color_cb)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.read_color_cam_info_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.read_depth_cb)
        rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.read_depth_cam_info_cb)
        rospy.Subscriber('/camera/points/xyzrgba', PointCloud2, self.read_point_cloud2_cb)
        self.raw_point_cloud2_msg = None

    def read_color_cb(self, msg):
        """Color image ROS subscriber callback.

        Args:
            msg (`sensor_msgs/Image`): Color image msg.
        """
        self.raw_color_img_msg = msg

    def read_depth_cb(self, msg):
        """Depth image ROS subcriber callback.

        Args:
            msg (`sensor_msgs/Image`): Depth image msg with `32FC1` or `16UC1`.
        """
        self.raw_depth_img_msg = msg

    def read_color_cam_info_cb(self, msg):
        """Color camera information subscriber callback.

        Args:
            msg (`sensor_msgs/CameraInfo`): Color camera information msg.
        """
        # msg.header.frame_id = self.camera_frame
        self.raw_color_cam_info_msg = msg

    def read_depth_cam_info_cb(self, msg):
        """Depth camera information ROS subscriber callback.

        Args:
            msg (`sensor_msgs/CameraInfo`): Depth camera information msg.
        """
        self.raw_depth_cam_info_msg = msg

    def read_point_cloud2_cb(self, msg):
        """Point cloud ROS subscriber callback.

        Args:
            msg (`sensor_msgs/PointCloud2`): Point cloud msg.
        """
        self.raw_point_cloud2_msg = msg

    @property
    def depth_img(self):
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if self.raw_depth_img_msg.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
        elif self.raw_depth_img_msg.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
            img = (img/1000.).astype(np.float32)

        # none to zero
        img = np.nan_to_num(img)

        # depth hole filling
        inpaint_mask = np.zeros(img.shape, dtype='uint8')
        inpaint_mask[img == 0] = 255
        restored_depth_image = cv2.inpaint(
            img,
            inpaint_mask,
            inpaintRadius=15,
            flags=cv2.INPAINT_NS
            )
        return restored_depth_image

    @property
    def depth_img_msg(self):
        return self.cv_bridge.cv2_to_imgmsg(self.depth_img)

    @property
    def not_inpainted_depth_img(self):
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if self.raw_depth_img_msg.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
        elif self.raw_depth_img_msg.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
            img = (img/1000.).astype(np.float32)

        # delete nan values
        img = np.nan_to_num(img)
        return img

    @property
    def not_inpainted_depth_img_msg(self):
        return self.cv_bridge.cv2_to_imgmsg(self.not_inpainted_depth_img)


    @property
    def color_img(self):
        """Color image from the subscribed color image topic.

        Returns:
            `numpy.ndarray`: (H, W, C) with `uint8` color image.
        """
        return self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "rgb8")

    @property
    def color_img_msg(self):
        return self.raw_color_img_msg

    @property
    def depth_cam_info_msg(self):
        """Depth camera information from the subscribed depth camera information topic.

        Returns:
            `sensor_msgs/CameraInfo`: Depth camera information msg.
        """
        return self.raw_depth_cam_info_msg

    @property
    def color_cam_info_msg(self):
        """Color camera information from the subscribed color camera information topic.

        Returns:
            `sensor_msgs/CameraInfo`: Color camera information msg.
        """
        return self.raw_color_cam_info_msg

    @property
    def color_cam_intr(self):
        """Color camera intrinsic matrix.

        Returns:
            numpy.ndarray: (3, 3) camera intrinsic matrix.
        """
        intr = np.array(self.color_cam_info_msg.K)
        return intr.reshape(3, 3)

    @property
    def depth_cam_intr(self):
        """Depth camera intrinsic matrix.

        Returns:
            numpy.ndarray: (3, 3) camera intrinsic matrix.
        """
        intr = np.array(self.depth_cam_info_msg.K)
        return intr.reshape(3, 3)

    @property
    def point_cloud2_msg(self):
        """Point cloud from the subscribed point cloud topic.

        Returns:
            `sensor_msgs/PointCloud2`: Point cloud msg.
        """
        # If camera does not publish point cloud, generate one from depth image
        if self.raw_point_cloud2_msg is None:
            rospy.logwarn('Point cloud is not published. Generate one from depth image.')
            pc2_msg = self._get_point_cloud2_msg(
                self.not_inpainted_depth_img,
                self.color_img,
                self.depth_cam_intr,
                self.depth_img_msg.header.frame_id)
            return pc2_msg
        return self.raw_point_cloud2_msg

    @property
    def xyz_img(self):
        """XYZ image from the depth image

        Returns:
            `numpy.ndarray`: (H, W, 3) with `float32` XYZ image.
        """
        pc_xyz = self._depth_to_point_cloud(
            self.depth_img,
            self.depth_cam_intr)
        pc_xyz = pc_xyz.astype(np.float32)
        H, W = self.depth_img.shape
        return pc_xyz[:, :3].reshape(H, W, 3)

    def _depth_to_point_cloud(self, depth_img, depth_cam_intr):
        """Convert depth image to point cloud.

        Args: 
            depth_img (`numpy.ndarray`): (H, W) with `float32` depth image.
            depth_cam_intr (`numpy.ndarray`): (3, 3) camera intrinsic matrix.

        Returns:
            `numpy.ndarray`: (HxW, 3) with `float32` point cloud.
        """
        height, width = depth_img.shape
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_img.flatten(), [3, 1])
        point_cloud = depth_arr * np.linalg.inv(depth_cam_intr).dot(pixels_homog)
        return point_cloud.transpose()

    def _get_point_cloud2_msg(self, depth_image, color_image, depth_cam_intr, camera_frame_id):
        """Visualize point cloud.

        Args:
            depth_image (`numpy.ndarray`): (H, W) with `float32` depth image.
            color_image (`numpy.ndarray`): (H, W, 3) with `uint8` color image.
            depth_cam_intr (`numpy.ndarray`): (3, 3) camera intrinsic matrix.
            camera_frame_id (`str`): camera frame id.
        """
        # get point cloud
        pc_xyz = self._depth_to_point_cloud(depth_image, depth_cam_intr)
        pc_rgb = color_image.reshape(-1,3)
        pc_bgra = np.zeros((pc_rgb.shape[0], 4), dtype=np.uint8)
        pc_bgra[:, :3] = pc_rgb[:, ::-1]

        # get point cloud message
        data = np.zeros(len(pc_xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'u4')])
        data['x'] = pc_xyz[:, 0]
        data['y'] = pc_xyz[:, 1]
        data['z'] = pc_xyz[:, 2]
        data['rgb'] = pc_bgra.view(dtype=np.uint32).reshape(-1)
        pc_msg = ros_numpy.point_cloud2.array_to_pointcloud2(data, frame_id=camera_frame_id)
        return pc_msg


class VisionSensorInterfaceScale(VisionSensorInterface):
    def __init__(self, scale_factor):
        super(VisionSensorInterfaceScale, self).__init__()
        self.scale_factor = scale_factor

    @property
    def color_img(self):
        """Color image from the subscribed color image topic.

        Returns:
            `numpy.ndarray`: (H, W, C) with `uint8` color image.
        """
        img = self.cv_bridge.imgmsg_to_cv2(self.raw_color_img_msg, "rgb8")
        return cv2.resize(img, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_NEAREST)

    @property
    def depth_img(self):
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if self.raw_depth_img_msg.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
        elif self.raw_depth_img_msg.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
            img = (img/1000.).astype(np.float32)

        # delete nan values
        img = np.nan_to_num(img)

        # resize
        resized_img = cv2.resize(
            img,
            None,
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST)

        # depth hole filling
        inpaint_mask = np.zeros(resized_img.shape, dtype='uint8')
        inpaint_mask[resized_img == 0] = 255
        restored_depth_image = cv2.inpaint(
            resized_img,
            inpaint_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_NS)
        self.inpaint_mask = inpaint_mask
        return restored_depth_image

    @property
    def not_inpainted_depth_img(self):
        """Depth image from the subscribed depth image topic.

        Returns:
            `numpy.ndarray`: (H, W) with `float32` depth image.
        """
        if self.raw_depth_img_msg.encoding == '32FC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
        elif self.raw_depth_img_msg.encoding == '16UC1':
            img = self.cv_bridge.imgmsg_to_cv2(self.raw_depth_img_msg)
            img = (img/1000.).astype(np.float32)

        # delete nan values
        img = np.nan_to_num(img)

        # resize
        resized_img = cv2.resize(
            img,
            None,
            fx=self.scale_factor,
            fy=self.scale_factor,
            interpolation=cv2.INTER_NEAREST)
        return resized_img

    @property
    def not_inpainted_depth_img_msg(self):
        return self.cv_bridge.cv2_to_imgmsg(self.not_inpainted_depth_img)

    @property
    def color_img_msg(self):
        return self.cv_bridge.cv2_to_imgmsg(self.color_img, "rgb8")

    @property
    def depth_img_msg(self):
        return self.cv_bridge.cv2_to_imgmsg(self.depth_img)

    @property
    def color_cam_info_msg(self):
        info = copy.deepcopy(self.raw_color_cam_info_msg)
        K = np.array(info.K)
        P = np.array(info.P)
        D = np.array(info.D)
        K[[0, 2, 4, 5]] *= self.scale_factor
        P[[0, 2, 5, 6]] *= self.scale_factor
        D *= self.scale_factor
        info.K = K
        info.P = P
        info.D = D
        info.height = int(info.height * self.scale_factor)
        info.width = int(info.width * self.scale_factor)
        return info

    @property
    def depth_cam_info_msg(self):
        info = copy.deepcopy(self.raw_depth_cam_info_msg)
        K = np.array(info.K)
        P = np.array(info.P)
        D = np.array(info.D)
        K[[0, 2, 4, 5]] *= self.scale_factor
        P[[0, 2, 5, 6]] *= self.scale_factor
        D *= self.scale_factor
        info.K = K
        info.P = P
        info.D = D
        info.height = int(info.height * self.scale_factor)
        info.width = int(info.width * self.scale_factor)
        return info
