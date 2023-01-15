#!/usr/bin/env python


import rospy
import numpy as np

from time import sleep
from copy import deepcopy

from robotiq_epick_control.msg import RobotiqEPick_robot_input
from robotiq_epick_control.msg import RobotiqEPick_robot_output


class RobotiqEpickGripperInterface(object):
    def __init__(self):
        self.gripper_status = None
        self.grasp_success = 1

        self.pub = rospy.Publisher(
            '/RobotiqEPickRobotOutput', RobotiqEPick_robot_output, queue_size=10)
        self.sub = rospy.Subscriber('/RobotiqEPickRobotInput',
                                    RobotiqEPick_robot_input, callback=self.gripper_status_cb)

        self.gripper_msg = RobotiqEPick_robot_output()

    def wait_for_connection(self, timeout=-1):
        rospy.sleep(0.1)
        r = rospy.Rate(30)
        start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if (timeout >= 0. and rospy.get_time() - start_time > timeout):
                return False
            if self.gripper_status is not None:
                return True
            r.sleep()
        return False

    def is_ready(self):
        return self.gripper_status.gFLT == 0 and self.gripper_status.gACT == 0

    def object_detected(self):
        return self.gripper_status.gOBJ == 1 or self.gripper_status.gOBJ == 2

    def get_fault_status(self):
        return self.gripper_status.gFLT

    def get_vacuum_lvl(self):
        "In % of absolute vacuum"
        return self.gripper_status.gPO
    
    def epick_advanced_mode(self, advanced_mode):
        if advanced_mode == True:
            self.gripper_msg.rMOD = 1
        else:
            self.gripper_msg.rMOD = 0
    
    def epick_advanced_mode_params(self, max_vacuum = 40, min_vacuum = 45, action_timeout = 20):
        self.gripper_msg.rPR = max_vacuum
        self.gripper_msg.rSP = action_timeout
        self.gripper_msg.rFR = min_vacuum

    def epick_activate_suction(self):
        """
        Activate the vacuum generation.
        """
        msg = deepcopy(self.gripper_msg)
        msg.rMOD = 1
        msg.rACT = 1
        msg.rATR = 0
        msg.rGTO = 1
        self.pub.publish(msg)

    def epick_release_vacuum(self):
        """
        Release the object to atmospheric pressure.
        """
        msg = deepcopy(self.gripper_msg)
        msg.rACT = 1
        msg.rATR = 1
        self.pub.publish(msg)

    def epick_deactivate_suction(self):
        """
        Deactivate vacuum generation. Does not necessarily relese an object.
        """
        msg = deepcopy(self.gripper_msg)
        msg.rACT = 1
        msg.rGTO = 0
        self.pub.publish(msg)

    def epick_reset_controller(self):
        """
        Resets the controller in case of error.
        """
        msg = deepcopy(self.gripper_msg)
        msg.rACT = 0
        msg.rGTO = 0
        self.pub.publish(msg)
        rospy.sleep(0.1)

    def gripper_status_cb(self, msg):
        self.gripper_status = msg




def main():
    rospy.init_node("robotiq_epick_ctrl_test")
    gripper = RobotiqEpickGripperInteface()
    print(gripper.wait_for_connection())
    if not gripper.is_ready():
        gripper.epick_reset_controller()

    gripper.epick_advanced_mode(True)
    gripper.epick_advanced_mode_params()


    while not rospy.is_shutdown():
        print("activating suction")
        gripper.epick_activate_suction()
        rospy.sleep(1)
        print(gripper.object_detected())
        gripper.epick_relese_vacuum()
        rospy.sleep(1)
        gripper.epick_reset_controller()
        rospy.sleep(1)



if __name__ == '__main__':
    main()
