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
ANGLE = 25.5  # Angle of the cameras from the ground plane
# SCALE = 1  # Scale of output of perspective transform image
IMAGE_WIDTH = 1280  # resolution of cameras - all should be the same
IMAGE_HEIGHT = 720


# SCALED_WIDTH = int(IMAGE_WIDTH * SCALE)  # Save recalculating every frame
# SCALED_HEIGHT = int(IMAGE_HEIGHT * SCALE)

# FIT_SCALE = 1  # This is just to tune the fitting of the images together


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

    return matrix

# Transform images to birds-eye perspective
# Then stitch images to get full picture
# This will be done using feature matching once
# The homography generated from this matching will then be used for all subsequent images
# This is to allow for 'real-time' processing

# We should build up the image in a loop
# We get the matching key points between the current + new image
# We then transform the new image onto the current
# This then repeats with the next image

def birds_eye_view(frames):
    matrix = get_transform_matrix()
    transformed = [cv.warpPerspective(frame, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT))
                   for frame in frames]

    transformed[2] = cv.rotate(transformed[2], cv.ROTATE_180)
    transformed[1] = cv.rotate(transformed[1], cv.ROTATE_90_COUNTERCLOCKWISE)
    transformed[3] = cv.rotate(transformed[3], cv.ROTATE_90_CLOCKWISE)

    sift = cv.SIFT.create(
        nfeatures=0,  # More features
        nOctaveLayers=5,  # More layers
        contrastThreshold=0.02,  # Lower for more features
        edgeThreshold=10  # Increased edge threshold
    )

    # Create large canvas and place first image 1/4 down
    canvas_size = (IMAGE_WIDTH * 6, IMAGE_HEIGHT * 6)
    result = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    x_offset = (canvas_size[0] - IMAGE_WIDTH) // 2
    y_offset = canvas_size[1] // 2
    result[y_offset:y_offset + IMAGE_HEIGHT,
    x_offset:x_offset + IMAGE_WIDTH] = transformed[2]
    current = result

    # This is just the order I want to do the test images in
    # but it could work in the default order
    order = [3, 1, 0]

    for i in order:
        new = transformed[i]

        # detect key points
        kp_current, desc_current = sift.detectAndCompute(current, None)
        kp_new, desc_new = sift.detectAndCompute(new, None)

        # Use BFMatcher for more accurate (but slower) matching
        matcher = cv.BFMatcher(cv.NORM_L2)
        matches = matcher.knnMatch(desc_current, desc_new, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # Stricter ratio
                good_matches.append(m)

        # Sort matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Take only the best matches
        good_matches = good_matches[:50]

        # This bit is only needed when you want to see the output of the matching
        #########################################################################
        #img_matches = cv.drawMatches(current, kp_current, new, kp_new, good_matches, None,
        #                             matchColor=(0, 255, 0),
        #                             singlePointColor=(255, 0, 0),
        #                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv.imshow("Matches", cv.resize(img_matches, (1280, 720)))
        #cv.waitKey(0)
        #########################################################################

        src_points = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
        dest_points = np.float32([kp_current[m.queryIdx].pt for m in good_matches])

        homography = cv.findHomography(src_points, dest_points, cv.RANSAC, 5.0)[0]
        warped = cv.warpPerspective(new, homography, canvas_size)

        mask1 = (current != 0).any(axis=2)
        mask2 = (warped != 0).any(axis=2)
        overlap = mask1 & mask2
        current[~mask1] = warped[~mask1]
        current[overlap] = cv.addWeighted(current[overlap], 0.5, warped[overlap], 0.5, 0)

        cv.imshow("stitched", cv.resize(current, (1280, 720)))
        cv.waitKey(0)

    # Final crop to content
    y_nonzero, x_nonzero = np.nonzero(current.any(axis=2))
    y_min, y_max = y_nonzero.min(), y_nonzero.max()
    x_min, x_max = x_nonzero.min(), x_nonzero.max()
    return current[y_min:y_max, x_min:x_max]

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
        # altered for testing with individual image
        #frames = [camera.retrieve()[1] for camera in cameras]
        for i in range(1, 5):
            frames.append(cv.imread(f'images/cam{i}.jpg'))

        # Undistort frames - better way to do this than loop?
        for i, frame in enumerate(frames):
            frames[i] = cv.remap(frame, x_matrices[i], y_matrices[i], cv.INTER_LINEAR)  # Not sure what best interp is

        # Stitch frame into birds eye image
        frame = birds_eye_view(frames)
        cv.imwrite("stitched_new.png", frame)
        cv.imshow("birds_eye", cv.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)))
        cv.waitKey(0)
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
