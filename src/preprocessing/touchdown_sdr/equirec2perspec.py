# Copyright: https://github.com/lil-lab/touchdown
import os
import sys
import cv2
import numpy as np

class Equirectangular:
    def __init__(self):
        # google street view setting
        # fov=110, height=900, width=1600
        # hfov = float(height) / width * wfov * 1.2
        pass

    def GetPerspective(self, image, fov, theta, phi, height, width, radius=128):
        # theta is left/right angle
        # phi is up/down angle, both in degree
        self._img = image
        self._height, self._width, _ = self._img.shape
        
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wfov = fov
        #hfov = float(height) / width * wfov #* 1.2
        hfov = 2 * np.degrees(np.arctan(float(height) / width * np.tan(np.radians(wfov/2))))

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wfov) / 2.0
        w_len = 2 * radius * np.sin(np.radians(wfov / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hfov) / 2.0
        h_len = 2 * radius * np.sin(np.radians(hfov / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)

        x_map = np.zeros([height, width], np.float32) + radius
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)

        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (radius / D * x_map)[:, :]
        xyz[:, :, 1] = (radius / D * y_map)[:, :]
        xyz[:, :, 2] = (radius / D * z_map)[:, :]
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / radius)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        
        persp = cv2.remap(
            self._img, 
            lon.astype(np.float32), 
            lat.astype(np.float32), 
            cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_WRAP
        )
        return persp
        
