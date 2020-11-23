##########################
# Function
##########################


def compute_descriptors(img, keypoints, feature_extractor):
    """[summary]

    Args:
        img ([type]): [description]
        keypoints ([type]): [description]
        feature_extractor ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Extract B, G, R
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 1]

    # Extract descriptors
    _, descriptors_b = feature_extractor.compute(img_b, keypoints)
    _, descriptors_g = feature_extractor.compute(img_g, keypoints)
    _, descriptors_r = feature_extractor.compute(img_r, keypoints)

    # Compute mean
    descriptors = (descriptors_b + descriptors_g + descriptors_r) / 3

    return descriptors
