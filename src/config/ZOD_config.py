import numpy as np

class_list = ['Vehicle', 'VulnerableVehicle', 'Pedestrian', 'Animal', 'StaticObject']

CLASS_NAME_TO_ID = {
    'Vehicle': 0,
    'VulnerableVehicle': 1,
    'Pedestrian': 2,
    'Animal': 3,
    'PoleObject': 4,
    'TrafficBeacon': 4,
    'TrafficSign': 4,
    'TrafficSignal': 4,
    'TrafficGuide': 4,
    'DynamicBarrier': 4,
    'Unclear': 4,
}


# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": -250,
    "maxX": 250,
    "minY": -250,
    "maxY": 250,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Back back (of vehicle) Point Cloud boundary for BEV
boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# TODO: Explore how to increase resolution
BEV_WIDTH = 608 # 608  # across y axis -25m ~ 25m
BEV_HEIGHT = 608 # 608  # across x axis 0m ~ 50m

# DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

DISCRETIZATION_X = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT 
DISCRETIZATION_Y = (boundary["maxY"] - boundary["minY"]) / BEV_WIDTH 

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
