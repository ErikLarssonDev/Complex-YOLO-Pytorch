import numpy as np
from rotate_iou import rotate_iou_gpu_eval


overlap_part = rotate_iou_gpu_eval(np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, 0]]),
                                   np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, 0]])).astype(np.float64)

# Testing, we expect to have 1.0 as result as the boxes are the same
print(f"overlap_part: {overlap_part}") # This works

overlap_part = rotate_iou_gpu_eval(np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, -1.4245373e-01]]),
                                   np.array([[6.0000977e+02, 8.1578074e+00,  2.0428976e+01,  5.0748917e+01, -1.4245373e-01]])).astype(np.float64)

# Testing, we expect to have 1.0 as result as the boxes are the same
print(f"overlap_part: {overlap_part}") # This does not work

# It seems like rotate_iou_gpu_eval does not work if the yaw is not 0.0