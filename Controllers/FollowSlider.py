import threading
import math
import time
from typing import List
import cv2
import asyncio
import numpy as np
import sympy as sym
#import RPi.GPIO as GPIO
import time

from HALs.HAL_base import HAL_base
#from Vision.VisionObject import VisionObject
#from Vision.VisualObjectIdentifier import VisualObjectIdentifier
from Modules.Base.ImageProducer import ImageProducer
from Controllers.Controller import Controller

def coordinate_input(x, y, z,hal,vision=False):
    global mtr
    global sim
    try:
        a1 = 5.3
        a2 = 3.25
        a3 = 11.4
        a4 = 3.25
        a5 = 5.3
        a6 = 5
        robot_arm1 = Three_Degree_Arm(a1, a2, a3, a4, a5, a6)
        # caluclate angles
        angles = robot_arm1.calculate_angles(sym.Matrix([x, y, z, 1]))

        if angles is None:
            EE = sym.Matrix([x, y, z, 1])
            x, y, z, t = EE
            if x != 0 or y != 0:
                theta = sym.atan2(y, x)
            else:
                theta = 0
            print(theta)
            opp = z - a1
            dist = sym.sqrt(x ** 2 + y ** 2)
            t0deg = float(sym.deg(theta).evalf()) + 180
            t0deg = (t0deg + 360) % 360
            if dist == 0:
                theta1 = 90
            else:
                theta1 = sym.atan(opp / dist)
            t1deg = 90 - (float(sym.deg(theta1).evalf()))
            t0n = hal.get_joint(0)

            t1n = hal.get_joint(1)
            t0n = t0n + (t0deg - t0n) * .1
            t1n = t1n + (t1deg - t1n) * .1
            hal.set_joint(0, t0deg)
            hal.set_joint(1, t1deg)
            hal.set_joint(2, 0)


        else:
            # ANGLES: angle[0]=base angle[1]=servo1 angle[2]=servo2

            # what is BASE ANGLE?

            EE = sym.Matrix([x, y, z, 1])
            x, y, z, t = EE
            if x != 0 or y != 0:
                theta = sym.atan2(y, x)
            else:
                theta = 0
            print(theta)
            opp = z - a1
            dist = sym.sqrt(x ** 2 + y ** 2)
            t0deg = float(sym.deg(theta).evalf()) + 180
            t0deg = (t0deg + 360) % 360
            angle_base = t0deg
            angle_base = (angle_base + 360) % 360

            # theta 1 output : -90 to 90
            angle_1 = -(angles[1]-90)

            # theta 2 output : 0 to 180
            angle_2 =-(angles[2]-90)

            print("--------- Moving ARM ---------")
            if vision:
                print(f"joint:{hal.get_joint(0)}, new: {angle_base}, mid: {hal.get_joint(0)+0.1*(angle_base-hal.get_joint(0))}")
                hal.set_joint(0,angle_base)
                hal.set_joint(1, hal.get_joint(1)+0.1*(angle_1-hal.get_joint(1)))
                hal.set_joint(2, hal.get_joint(2)+0.1*(angle_2-hal.get_joint(2)))
            else:
                hal.set_joint(0, angle_base)
                hal.set_joint(1, angle_1)
                hal.set_joint(2, angle_2)

    except Exception as err:
        print(f"Exeption in moving the arm (coordinate_input): {err=}, {type(err)=}")


class Three_Degree_Arm:
    def __init__(self, a1, a2, a3, a4, a5, a6) -> None:
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6

    def DH_matrix(self, theta, alpha, r, d):
        return sym.Matrix([
            [sym.cos(theta), -sym.sin(theta) * sym.cos(alpha), sym.sin(theta) * sym.sin(alpha), r * sym.cos(theta)],
            [sym.sin(theta), sym.cos(theta) * sym.cos(alpha), -sym.cos(theta) * sym.sin(alpha), r * sym.sin(theta)],
            [0, sym.sin(alpha), sym.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def calculate_angles(self, EE):
        x, y, z = EE[0], EE[1], EE[2]

        # Condition checks
        cond1 = (np.abs(x) - self.a3) ** 2 + (z - self.a1) ** 2 < (self.a5 + self.a6) ** 2 and np.abs(
            x) > self.a3 and 0 < z < self.a1
        cond2 = np.abs(x) ** 2 + (z - self.a1) ** 2 < (self.a3 + self.a5 + self.a6) ** 2 and np.abs(x) ** 2 + (
                z - self.a1) ** 2 > self.a3 ** 2 and np.abs(x) > 0 and z > self.a1

        isInRegion = cond1 or cond2

        if isInRegion:
            print('The point is inside the region.')
            print(EE)
            theta1 = sym.atan2(y, x)

            H0_1 = self.DH_matrix(theta1, sym.pi / 2, 0, self.a1)
            EE_local_H0_1 = H0_1.inv() * EE

            EE_x_H0_1 = EE_local_H0_1[0]
            # not used in this context
            EE_y_H0_1 = EE_local_H0_1[2]
            EE_z_H0_1 = EE_local_H0_1[1]

            l1 = self.a3
            l2 = self.a5 + self.a6
            R = sym.sqrt(EE_x_H0_1 ** 2 + EE_z_H0_1 ** 2)
            theta = sym.atan(EE_z_H0_1 / EE_x_H0_1)

            beta = sym.acos((R ** 2 - l1 ** 2 - l2 ** 2) / (-2 * l1 * l2))
            if not beta.is_real:
                return None
            alpha = sym.asin((l2 * sym.sin(beta)) / R)
            print(alpha)
            theta2 = theta + alpha
            theta3 = beta - sym.pi / 2

            theta4 = 0

            print(theta2.evalf())
            print(theta3.evalf())

            H1_2 = self.DH_matrix(theta2, 0, self.a3, -self.a2)
            H2_3 = self.DH_matrix(theta3, sym.pi / 2, 0, self.a4)
            H3_4 = self.DH_matrix(theta4, 0, 0, self.a5 + self.a6)

            H0_2 = H0_1 * H1_2
            H0_3 = H0_2 * H2_3
            H0_4 = H0_3 * H3_4

            EE_position = H0_4[:3, 3]
            print('EE Position (CM):')
            # print(f'x: {float(EE_position[0].evalf()):.2f}')
            # print(f'y: {float(EE_position[1].evalf()):.2f}')
            # print(f'z: {float(EE_position[2].evalf()):.2f}')

            theta1_deg = sym.deg(theta1.evalf())
            theta2_deg = sym.deg(theta2.evalf())
            theta3_deg = sym.deg(theta3.evalf())

            print('\nTheta Angles:')
            print(f'Base: {theta1_deg.evalf():.2f} degrees')
            print(f'1: {theta2_deg.evalf():.2f} degrees')
            print(f'2: {theta3_deg.evalf():.2f} degrees')
            return theta1_deg, theta2_deg, theta3_deg
        else:
            print('This point is not feasible')

class FollowSliderController(Controller):
    def __init__(self, selected_HAL: HAL_base):
        self.selected_HAL: HAL_base = selected_HAL
        #self.vision: VisualObjectIdentifier = vision
        #self.imageGetter: ImageProducer = selected_HAL
        
        self._task = None  # To keep track of the running task
        self.keep_running = False
        self.thread = None
        self.verbose_logging = False
        
        self.init = False
        self.frame=False
        self.mask=False
        self.paused = False
        # in radians
        self.contours = False
        selected_HAL.set_joint_min(0, 0) # set_base_min_degree(0)
        selected_HAL.set_joint_max(0, 270) # set_base_max_degree(270)
        selected_HAL.set_joint_max(2, 75) # set_joint_2_max(75)
        
  
        
    def thread_main(self):
        pass

    def start(self):
        self.keep_running = True
        self.paused = True
        xEE = 1
        yEE = 0
        zEE = 17
        Incrementing = True
        counter = 0
        range = (5*4) #a * b is how many times it will increment up then down(changing b depending on factor ex: 4 if icrementing by .25)
        while(True): #loop to keep moving back and forth in one axis
            coordinate_input(xEE,yEE,zEE,self.selected_HAL,False) #change end effector ie move the arm to specified cordinate
            if(Incrementing): 
                xEE += .25
            else:
                xEE -= .25
            counter += 1
            if (counter >= range):
                counter = 0
                Incrementing = not(Incrementing)
            time.sleep(.25) #added a delay
        #self.thread = threading.Thread(target=self.thread_main)
        #self.thread.start()
    def pause_state(self,state):
        self.paused = state
    
    def stop(self):
        self.keep_running = False