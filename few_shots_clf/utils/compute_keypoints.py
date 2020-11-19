##########################
# Imports
##########################


# Global
import cv2


##########################
# Function
##########################


def compute_keypoints(img, kpt_stride, kpt_sizes):
    """[summary]

    Args:
        img ([type]): [description]
        kpt_stride ([type]): [description]
        kpt_sizes ([type]): [description]

    Returns:
        [type]: [description]
    """
    return [
        cv2.KeyPoint(x, y, size)
        for size in kpt_sizes
        for x in range(0, img.shape[1], kpt_stride)
        for y in range(0, img.shape[0], kpt_stride)
    ]
