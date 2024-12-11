import numpy as np
import cv2 as cv
import os

# Perspective transform
VERTICAL_CROP = 1 / 2  # What portion of the vertical resolution of the images to use
ANGLE = 23.5 # Angle of the cameras from the ground plane
# SCALE = 1  # Scale of output of perspective transform image
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# This script generates the homography matrices for the cameras
# This is currently done by using one set of images to generate the homography
# This is then saved for use in the main script
# Possibility to use multiple frames and average the homography matrices
# Do this over each frame from a video and then average
# Randomise the order of the matching + stitching each time to get a better average

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


def calculate_homographes(frames):

    matrix = get_transform_matrix()

    transformed = [cv.warpPerspective(frame, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT))
                   for frame in frames]

    sift = cv.SIFT.create(
        nfeatures=0,  # More features
        nOctaveLayers=5,  # More layers
        contrastThreshold=0.01,  # Lower for more features
        edgeThreshold=10  # Increased edge threshold
    )

    # Create large canvas and place first image at the top middle
    canvas_size = (int(IMAGE_WIDTH * 1.6), int(IMAGE_HEIGHT * 2.75))
    result = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Change y_offset to 0 to place at top
    x_offset = (canvas_size[0] - IMAGE_WIDTH) // 2
    y_offset = 0  # Changed from canvas_size[1] // 2 to 0

    result[y_offset:y_offset + IMAGE_HEIGHT,
           x_offset:x_offset + IMAGE_WIDTH] = transformed[2]
    current = result

    cv.imshow("stitched", cv.resize(current, (1280, 720)))
    cv.waitKey(0)

    # This is just the order I want to do the test images in
    # but it could work in the default order
    order = [1, 0, 3]

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
        """
        #########################################################################
        img_matches = cv.drawMatches(current, kp_current, new, kp_new, good_matches, None,
                                     matchColor=(0, 255, 0),
                                     singlePointColor=(255, 0, 0),
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("Matches", cv.resize(img_matches, (1280, 720)))
        cv.waitKey(0)
        #########################################################################
        """

        src_points = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
        dest_points = np.float32([kp_current[m.queryIdx].pt for m in good_matches])

        homography = cv.findHomography(src_points, dest_points, cv.RANSAC, 5.0)[0]
        warped = cv.warpPerspective(new, homography, canvas_size)

        mask1 = (current != 0).any(axis=2)
        current[~mask1] = warped[~mask1]
        mask1 = mask1.astype(np.uint8) * 255
        np.save(f"masks/mask{i}", ~mask1)

        cv.imshow("stitched", cv.resize(current, (1280, 720)))
        cv.waitKey(0)
        homography = homography.astype(np.float64)
        np.save(f'homographes/homography{i}', homography)

    # Final crop to content
    y_nonzero, x_nonzero = np.nonzero(current.any(axis=2))
    y_min, y_max = y_nonzero.min(), y_nonzero.max()
    x_min, x_max = x_nonzero.min(), x_nonzero.max()
    cv.imwrite("homographes/result1.png", current[y_min:y_max, x_min:x_max])
    x = np.array([x_min, x_max])
    y = np.array([y_min, y_max])

    np.savez("masks/corners", x=x, y=y)

def main():

    x_matrices = []
    y_matrices = []
    for file in os.listdir("matrices"):
        mats = np.load("matrices/" + file)
        x_matrices.append(mats['mapx'])
        y_matrices.append(mats['mapy'])

    frames = [cv.imread(f'images/cam{i+1}.jpg') for i in range(4)]

    for i, frame in enumerate(frames):
        frames[i] = cv.remap(frame, x_matrices[i], y_matrices[i], cv.INTER_LINEAR)

    calculate_homographes(frames)


if __name__ == "__main__":
    main()
