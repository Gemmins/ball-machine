"""
Vision steps:
- read in video from each of the 4 camera streams
- undistort the image using the precomputed matrix and cv.remap()
- perform a perspective transform and stitch the images to get a birds-eye representation
- (could possibly combine the undistortion matrix?)
- blur image and extract lines
- using the location of lines, find the best location of the tennis court
- calculate location of the robot relative to the tennis court
"""
import os
import re
from idlelib.pyparse import trans

import numpy as np
import cv2 as cv

# These are the video capture indexes of the cameras I'm using
# I've numbered the cameras externally, so they are in that order
CAMERAS = [7, 2, 1, 6]

# Gaussian Blur
KERNEL_SIZE = (5, 5)  # Kernel size
SIGMA_X = 1  # Sigma

# Perspective transform
VERTICAL_CROP = 1 / 2  # What portion of the vertical resolution of the images to use
ANGLE = 23.5  # Angle of the cameras from the ground plane
# SCALE = 1  # Scale of output of perspective transform image
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def get_transform_matrix():
    image_width = IMAGE_WIDTH
    image_height = IMAGE_HEIGHT

    centre_x = image_width / 2
    expansion = 1 / np.cos(np.deg2rad(90 - ANGLE))
    bottom_width = image_width * expansion
    bottom_offset = image_height * (1 - VERTICAL_CROP)

    src_points = np.float32([[0, bottom_offset],
                             [image_width, bottom_offset],
                             [centre_x - (bottom_width / 2), image_height],
                             [centre_x + (bottom_width / 2), image_height]])

    # should probs use smaller resolution than source image
    dest_points = np.float32([[0, bottom_offset],
                              [image_width, bottom_offset],
                              [0, image_height],
                              [image_width, image_height]])

    matrix = cv.getPerspectiveTransform(src_points, dest_points)

    return matrix.astype(np.float64)


# The image is stitched together using precomputed homography matrices
# The image masks and canvas size will also be precomputed as to save time

def stitch(frames, matrix, homographes, masks, x, y):

    frames = [cv.warpPerspective(frame, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT)) for frame in frames]

    # Create large canvas and place first image at the top middle
    # How this bit is done will be changed but is fine for now
    canvas_size = (int(IMAGE_WIDTH * 1.6), int(IMAGE_HEIGHT * 2.75))
    result = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Change y_offset to 0 to place at top
    x_offset = (canvas_size[0] - IMAGE_WIDTH) // 2
    y_offset = 0  # Changed from canvas_size[1] // 2 to 0

    roi = result[y_offset:y_offset + IMAGE_HEIGHT, x_offset:x_offset + IMAGE_WIDTH]
    cv.copyTo(frames[2], None, roi)
    current = result

    # This is just the order I want to do the test images in
    # but it could work in the default order
    order = [1, 0, 3]

    for i in order:
        homography = homographes[i]
        warped = cv.warpPerspective(frames[i], homography, canvas_size)  # Want to combine warps but it's not working

        # Convert mask to 3-channel format for bitwise operations
        mask_3ch = cv.cvtColor(masks[i].astype(np.uint8), cv.COLOR_GRAY2BGR)

        # Perform the stitching using bitwise operations
        current = cv.bitwise_and(warped, mask_3ch) + cv.bitwise_and(current, ~mask_3ch)

    # Final crop to content
    return current[y[0]:y[1], x[0]:x[1]]


# TODO
def detect_lines(frame):
    lines = []
    return lines


# TODO
def fit_court(corners):
    return 0, 0, 0


# TODO
def estimate_location(x, y, angle):
    return x, y


def main():
    cameras = []
    for i in CAMERAS:
        cameras.append(cv.VideoCapture(i))
    for i, camera in enumerate(cameras):
        if not camera.isOpened():
            print(f"Cannot open camera {i}")
            #exit()

    # load the pre-calculated calibration matrices for the cameras
    # these should already be in the correct order for the cameras
    x_matrices = []
    y_matrices = []
    for file in os.listdir("matrices"):
        mats = np.load("matrices/" + file)
        x_matrices.append(mats['mapx'])
        y_matrices.append(mats['mapy'])

    frames = []

    matrix = get_transform_matrix()

    corners = np.load("masks/corners.npz")
    x, y = corners['x'], corners['y']

    # Load the homographes for the cameras
    # Done in this way so order of matrix in dir is irrelevant
    # TODO sort this out
    homographes = [None] * 4
    masks = [None] * 4
    for file in os.listdir("homographes"):
        if "npy" in file:
            num = re.search(r'\d+', file).group()
            homographes[int(num)] = np.load("homographes/" + file)
            masks[int(num)] = np.load("masks/mask" + num + ".npy")

    while True:
        # Capture frame-by-frame - grab and then retrieve for better sync
        # I think I might eventually use multiprocessing to get better sync but im not sure
        for camera in cameras:
            camera.grab()
        # altered for testing with individual image
        # frames = [camera.retrieve()[1] for camera in cameras]
        for i in range(1, 5):
            frames.append(cv.imread(f'images/cam{i}.jpg'))

        # Undistort frames
        frames = [cv.remap(frame, x_matrices[i], y_matrices[i], cv.INTER_LINEAR) for i, frame in enumerate(frames)]
        # for i, frame in enumerate(frames):
        #    frames[i] = cv.remap(frame, x_matrices[i], y_matrices[i], cv.INTER_LINEAR)  # Not sure what best interp is

        # Stitch frame into birds eye image
        frame = stitch(frames, matrix, homographes, masks, x, y)
        cv.imwrite("stitched_new.png", frame)
        cv.imshow("birds_eye", cv.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)))
        cv.waitKey(0)
        break
        # Extract corners from birds eye image
        corners = detect_lines(frame)

        # Estimate location of tennis court in the frame using corners
        x, y, angle = fit_court(corners)

        # Using location of court - calculate location of robot
        estimate_location(x, y, angle)

        # TODO
        # Ideally output visualisation of tennis court with position estimate on it
        # Side by side with birds eye view

        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    for camera in cameras:
        camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
