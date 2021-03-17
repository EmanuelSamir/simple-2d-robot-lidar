#!/usr/bin/python

import numpy as np
from .utils import *
import time
from matplotlib import pyplot as plt


class Robot2D:
    def __init__(self,
                env_max_size = 5,  
                env_min_size = -5, 
                robot_radius = 0.4,
                lidar_max_range = 2., 
                dT = 0.01, 
                is_render = True):

        # Environment
        self.env = Environment(env_min_size, env_max_size)
        self.env_min_size = env_min_size    
        self.env_max_size = env_max_size    

        # Initial State
        self.xr = 0
        self.yr = 0
        self.rr = robot_radius

        # Lidar parameters
        self.max_range = lidar_max_range
        self.xls = []
        self.yls = []

        # Parameters for simulation
        self.dT = dT
        self.is_render = is_render
        self.fig = None
        self.ax = None


    def reset(self):
        self.xr = np.random.uniform(low = self.env_min_size + self.rr , high = self.env_max_size - self.rr)
        self.yr = np.random.uniform(low = self.env_min_size + self.rr , high = self.env_max_size - self.rr)

        self.env.get_random_obstacles(self.xr, self.yr, self.rr)
        self.xls = []
        self.yls = []

        # If render enabled, 
        if self.is_render:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10,10))
            self.ax.set_xlim((-5, 5))
            self.ax.set_ylim((-5, 5))
            circle = plt.Circle((self.xr, self.yr), self.rr, color='r', fill=True)
            self.ax.add_patch(circle)
            plt.pause(0.5)

    def set_init_state(self, x0, y0):
        self.xr = x0
        self.yr = y0

        cond = False
        if ( (self.xr > self.env_max_size - self.rr ) or (self.xr < self.env_min_size + self.rr )):
            raise ValueError('x value: {} is out of range {} and {}'.format(self.xr, self.env_min_size, self.env_max_size))

        if ( (self.yr > self.env_max_size - self.rr ) or (self.yr < self.env_min_size + self.rr )):
            raise ValueError('y value: {} is out of range {} and {}'.format(self.yr, self.env_min_size, self.env_max_size))

        self.env.get_random_obstacles(self.xr, self.yr, self.rr)
        self.xls = []
        self.yls = []

        # If render enabled, 
        if self.is_render:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10,10))
            self.ax.set_xlim((-5, 5))
            self.ax.set_ylim((-5, 5))
            circle = plt.Circle((self.xr, self.yr), self.rr, color='r', fill=True)
            self.ax.add_patch(circle)
            plt.pause(0.5)
            

    def step(self, vx, vy):
        self.xr = self.xr + self.dT * vx
        self.yr = self.yr + self.dT * vy
 
    def is_crashed(self):
        xcs = self.env.xcs
        ycs = self.env.ycs
        rcs = self.env.rcs

        if ( (self.xr > self.env_max_size - self.rr ) or (self.xr < self.env_min_size + self.rr )):
            return True

        if ( (self.yr > self.env_max_size - self.rr ) or (self.yr < self.env_min_size + self.rr )):
            return True

        for xc, yc, rc in zip(xcs, ycs, rcs):
            if ( np.sqrt( (xc - self.xr ) ** 2 + (yc - self.yr) ** 2) ) < rc + self.rr:
                return True
        return False


    def scanning(self):
        xcs = self.env.xcs
        ycs = self.env.ycs
        rcs = self.env.rcs

        r = self.max_range
        self.xls = []
        self.yls = []
        ths = np.arange(0,360, 4) 
        for th in ths:
            thr = np.deg2rad(th)
            self.xls.append(self.xr + r * np.cos(thr))
            self.yls.append(self.yr + r * np.sin(thr))  
         
        for i, (xl, yl, th) in enumerate(zip(self.xls, self.yls, ths)):
            for xc, yc, rc in zip(xcs, ycs, rcs):   
                is_inter, result = obtain_intersection_points(self.xr, self.yr, xl, yl, xc, yc, rc) 
                if is_inter:
                    cond = validate_point(result[0] - self.xr, result[1] - self.yr, self.xls[i] - self.xr, self.yls[i] - self.yr, th, self.max_range)
                    if cond:
                        self.xls[i] = result[0]
                        self.yls[i] = result[1]


    def render(self):
        if self.is_render:
            xcs = self.env.xcs
            ycs = self.env.ycs
            rcs = self.env.rcs
            self.ax.clear()
            self.ax.set_xlim((-5, 5))
            self.ax.set_ylim((-5, 5))
            circle = plt.Circle((self.xr, self.yr), self.rr, color='r', fill=True,zorder=10)
            self.ax.add_patch(circle)
            self.ax.scatter( self.xls, self.yls , color = 'r')
            for xc, yc, rc in zip(xcs, ycs, rcs):
                circle = plt.Circle((xc, yc), rc, color='b', fill=True)
                self.ax.add_patch(circle)
            for xl, yl in zip(self.xls, self.yls):
                self.ax.plot([self.xr, xl] ,[self.yr, yl], color = 'gray')
            
            plt.pause(0.02) 
            self.fig.canvas.draw()

    def close(self):
        plt.ioff()
        plt.close()




class Environment:
    def __init__(self, env_min_size, env_max_size, xcs  = np.array([]), ycs  = np.array([]), rcs  = np.array([])):
        # env param
        self.env_min_size = env_min_size
        self.env_max_size = env_max_size
        
        # obstacles 3- x, y, radius        print(self.xcs)
        self.xcs = xcs 
        self.ycs = ycs
        self.rcs = rcs

    def _random_point_without_robot(self, pr, rr, r):
        cond = False
        while not cond:
            p = np.random.uniform(low = self.env_min_size + rr + r, high = self.env_max_size - r -rr) 
            if ( (p < pr + r + rr )  and (p > pr - r - rr) ):
                pass
            else:
                cond = True
        return p


    def get_random_obstacles(self, xr, yr, rr, n = 30, r = 0.3):
        xcs = []
        ycs = []
        rcs = n * [r]
        
        for _ in range(n):
            xcs.append(self._random_point_without_robot(xr, rr, r))
            ycs.append(self._random_point_without_robot(yr, rr, r))

        self.xcs = np.array(xcs)
        self.ycs = np.array(ycs)
        self.rcs = np.array(rcs)

