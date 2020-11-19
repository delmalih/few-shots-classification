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
    _, descriptors = feature_extractor.compute(img, keypoints)
    return descriptors
