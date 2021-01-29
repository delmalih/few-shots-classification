##########################
# Imports
##########################


import os
from typing import List


##########################
# Function
##########################


def get_all_images_from_folder(path: str) -> List[str]:
    """Lists all image paths contained in a folder.

    Args:
        path (str): The given folder path.

    Returns:
        List[str]: The list of image paths in the folder.
    """
    # Init paths list
    image_paths = []

    # Loop over all sub-folders and sub-files
    for root, _, files in os.walk(path):

        # Loop over all sub-files
        for img_filename in files:
            # Get final file path
            file_path = os.path.join(root, img_filename)

            # Get extension of the file
            _, file_ext = os.path.splitext(file_path)

            # Attempt to read image and
            # add its path to the list
            if is_readable_by_opencv(file_ext):
                image_paths.append(file_path)

    # Sort paths
    image_paths = sorted(image_paths)

    return image_paths


def is_readable_by_opencv(extension: str) -> bool:
    """Returns if an image extension is readable by opencv

    Args:
        extention (str): The extenstion of the image

    Returns:
        bool: true if it is readable by opencv
    """
    opencv_readable_extensions = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe",
                                  ".jp2", ".png", ".pbm", ".pgm", ".ppm",
                                  ".sr", ".ras", ".tiff", ".tif"]
    extension = extension.lower()
    return extension in opencv_readable_extensions
