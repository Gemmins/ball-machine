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

import numpy as np
import cv2 as cv

# These are the video capture indexes of the cameras I'm using
# I've numbered the cameras externally, so they are in that order
CAMERAS = [7, 2, 1, 6]

# Gaussian Blur
KERNEL_SIZE = (5, 5)  # Kernel size
SIGMA_X = 1  # Sigma

# Corner detection
BLOCK_SIZE = 7  # Neighborhood size
K_SIZE = 3  # Aperture parameter for the Sobel operator
K = 0.04  # Harris detector free parameter
THRESHOLD = 0.01


# TODO
# Transform images to birds-eye perspective
# Then stitch images to get full picture
# Can probably do this buy hand as feature detection for stitching may take too long
# And the cameras dont change position relative to each other
def birds_eye_view(frames):
    return np.zeros((1, 1))


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
            exit()

    # load the pre-calculated calibration matrices for the cameras
    # these should already be in the correct order for the cameras
    x_matrices = []
    y_matrices = []
    for file in os.listdir("x_matrices"):
        x_matrices.append(np.load("x_matrices/" + file))
    for file in os.listdir("x_matrices"):
        y_matrices.append(np.load("y_matrices/" + file))

    while True:
        # Capture frame-by-frame - grab and then retrieve for better sync
        # I think I might eventually use multiprocessing to get better sync but im not sure
        for camera in cameras:
            camera.grab()

        frames = [camera.retrieve()[1] for camera in cameras]

        # Undistort frames
        frames = map(lambda frame: cv.remap(frame, x_matrices[i], y_matrices[i]), frames)  # IDK if this one works yet
        #new_frames = [cv.remap(frame, xmap, ymap) for frame in frames for xmap in x_matrices for ymap in y_matrices]

        # Stitch frame into birds eye image
        frame = birds_eye_view(frames)

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
