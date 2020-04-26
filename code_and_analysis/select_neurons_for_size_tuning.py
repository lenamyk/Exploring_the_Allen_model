#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters
----------
radius_limit: radius from cortex center (micrometer)
"""


import h5py
import numpy as np


radius_limit = 70


f = h5py.File('./v1_nodes.h5','r')
x = f['/nodes/v1/0/x']
z = f['/nodes/v1/0/z']
ids = f['/nodes/v1/node_id']
xcoord = np.asarray(x)
zcoord = np.asarray(z)
ids = np.asarray(ids)


radius = np.add((xcoord**2),(zcoord**2))
radius = np.sqrt(radius)


# Find center neuron:
print(np.argmin(radius))  


# Find neurons within a specified radius of the center:
close_to_center = np.flatnonzero(radius < radius_limit)             
np.savetxt('70microm_to_center_ids.txt', close_to_center, fmt='%s')
