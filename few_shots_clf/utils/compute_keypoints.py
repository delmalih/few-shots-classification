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
    # Init keypoints
    keypoints = []

    # Loop over size
    for size in kpt_sizes:
        # Loop over x
        for x_coord in range(0, img.shape[1] - int(size / 2), kpt_stride):
            # Loop over y
            for y_coord in range(0, img.shape[0] - int(size / 2), kpt_stride):
                kpt = cv2.KeyPoint(x_coord, y_coord, size)
                kpt.class_id = 0
                keypoints.append(kpt)

    return keypoints
