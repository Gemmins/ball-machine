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

# Perspective transform
VERTICAL_CROP = 1 / 2  # What portion of the vertical resolution of the images to use
ANGLE = 90  # Angle of the cameras from the ground plane
SCALE = 1  # Scale of output of perspective transform image
IMAGE_WIDTH = 1280  # resolution of cameras - all should be the same
IMAGE_HEIGHT = 720

SCALED_WIDTH = int(IMAGE_WIDTH * SCALE)  # Save recalculating every frame
SCALED_HEIGHT = int(IMAGE_HEIGHT * SCALE)

FIT_SCALE = 0.9  # This is just to tune the fitting of the images together


def get_transform_matrix():
    image_width = IMAGE_WIDTH
    image_height = IMAGE_HEIGHT

    centre_x = image_width / 2

    #expansion = 1 / np.cos(np.deg2rad(90-ANGLE))
    expansion = 1.6
    bottom_width = image_width * expansion

    src_points = np.float32([[0, image_height * (1-VERTICAL_CROP)],
                             [image_width, image_height * (1-VERTICAL_CROP)],
                             [centre_x - bottom_width, image_height],
                             [centre_x + bottom_width, image_height]])

    # should probs use smaller resolution than source image
    dest_points = np.float32([[0, 0],
                              [image_width * SCALE, 0],
                              [0, image_height * SCALE],
                              [image_width * SCALE, image_height * SCALE]])

    matrix = cv.getPerspectiveTransform(src_points, dest_points)

    return matrix


# TODO
# Transform images to birds-eye perspective
# Then stitch images to get full picture
# Can probably do this buy hand as feature detection for stitching may take too long
# And the cameras dont change position relative to each other
def birds_eye_view(frames, matrix):
    frames = [cv.warpPerspective(frame, matrix, (SCALED_WIDTH, SCALED_HEIGHT)) for frame in frames]

    new_height = int(SCALED_HEIGHT * FIT_SCALE)
    frames = [cv.resize(frame, (SCALED_WIDTH, new_height)) for frame in frames]

    # Rotate to fit
    frames[1] = cv.rotate(frames[1], cv.ROTATE_90_COUNTERCLOCKWISE)
    frames[2] = cv.rotate(frames[2], cv.ROTATE_180)
    frames[3] = cv.rotate(frames[3], cv.ROTATE_90_CLOCKWISE)

    for i, frame in enumerate(frames):
        cv.imwrite(f'output/transformed-{i}.png', frame)

    return 1


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

    frames = [] * 4

    matrix = get_transform_matrix()

    while True:
        # Capture frame-by-frame - grab and then retrieve for better sync
        # I think I might eventually use multiprocessing to get better sync but im not sure
        for camera in cameras:
            camera.grab()

        #frames = [camera.retrieve()[1] for camera in cameras]
        for i in range(1, 5):
            frames.append(cv.imread(f'images/cam{i}.jpg'))

        # Undistort frames - better way to do this than loop?
        for i, frame in enumerate(frames):
            frames[i] = cv.remap(frame, x_matrices[i], y_matrices[i], cv.INTER_LINEAR)  # Not sure what best interp is

        # Stitch frame into birds eye image
        frame = birds_eye_view(frames, matrix)

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
