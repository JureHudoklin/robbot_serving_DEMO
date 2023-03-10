#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class Config(object):
    AUTO_EXECUTE = False
    SERVE_SPEED_SCALE = 2.0
    FAST_SPEED_SCALE = 2.0

    # Move Group Config
    CUP_GROUP_NAME = 'irb120_cup'
    PLATE_GROUP_NAME = 'irb120_plate'
    BASE_NAME = 'irb120_base'

    # TF NAME
    MOBILE_ID = 'mobile_robot'
    CUP_PLACE_ID = 'cup_place'
    PLATE_PLACE_ID = 'plate_place'  

    # Standby joint configuratio 
    CUP_STANDBY_JOINT_CONFIG = [0.04657674,  0.03858206,  0.55117667,  1.59680092, - 1.60959554, - 2.55120587]
    # PLATE_STANDBY_JOINT_CONFIG = [8.460736717097461e-05, -0.07979149371385574,
    #                               0.6024545431137085, -0.00013575352204497904, -0.5229216814041138, -1.5708903074264526]

    PLATE_STANDBY_JOINT_CONFIG = [2.3433198293787427e-05, 0.0245346836745739, 0.707938551902771,
        0.00012810883345082402, -0.732477605342865, -1.5707485675811768]
    PLATE_PLACE_JOINT_CONFIG = [-2.225698709487915, -0.3565397560596466,
                                0.07542303949594498, 0.5986678004264832, 1.086991310119629, -0.7220945954322815]

    CUP_0_POSE_POS = [0.55, 0, 0.41]
    CUP_0_POSE_ORI = [-0.5000364328180643,
                      0.4999328709664223, -0.4999930679447441, 0.5000376209737208]
    PLATE_0_POSE_POS = [0.66, -0.0519629675569066, 0.41]
    PLATE_0_POSE_ORI = [-0.5000465529211938,
                        0.49991546337347875, -0.4999903985523663, 0.500047573483921]

    ZERO_POSE = [0.55, 0, 0.41]
    ZERO_POSE_JOINTS = [-6.0260703321546316e-05, 0.024607492610812187, 0.7079266309738159, -0.0004413560382090509, -0.7326934337615967, -1.570569634437561]

    # Standby Pose(Based on cup pose)
    CUP_STANDBY_POS = [0.300, -0.258, 0.450]
    CUP_STANDBY_ORI = [0.0, 0.7071068, -0.7071068, 0.0]
    PLATE_STANDBY_POS = [0.248, -0.367, 0.450]
    PLATE_STANDBY_ORI = [0.0, -0.7071068, 0.7071068, 0.0]

    # Difference between table and robot base: z-130mm
    # Plate Grasp Pose(frame 'irb120_base')
    PLATE_PICK_POS = [-0.1075, -0.455, 0.325]  # ( (-) y --> go deeper)
    PLATE_PICK_ORI = [0.6272114, 0.6272114, -0.3265056, 0.3265056]  # e_xyz: (125, 0, -90)
    CUP_PICK_POS = [0.300, -0.630, 0.230]  # (z: cup height 90 mm)
    CUP_PICK_ORI = [0.0, 0.7071068, -0.7071068, 0.0]  # e_xyz: (90, 0, -180)

    # Middle pose for the B-Spline
    PLATE_MIDDLE_POS = [0.5, -0.2, 0.4]
    CUP_MIDDLE_POS = [0.6, 0.0, 0.5]

    # Place pose(frame 'mobile')
    #CUP_PLACE_POS = [0.0, 0.0, 0.15]
    CUP_PLACE_POS = [-0.120, 0.06, 0.09]
    CUP_PLACE_ORI = [-0.7071068, 0.0, 0.0, 0.7071068] # e_xyz: (-90, 0, 0)
    PLATE_PLACE_POS = [0.100, 0.08, 0.05]
    PLATE_PLACE_ORI = [-0.6272114, 0.6272114, 0.3265056, 0.3265056]  # e_xyz: (-125, 0, -90)

    # Offsets
    CUP_PICK_Z_OFFSET = 0.1
    PLATE_PICK_APP_OFFSET = -0.1
    PLATE_PICK_Z_OFFSET = 0.1
