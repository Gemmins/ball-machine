import sys
import math
import cv2 as cv
import numpy as np
from itertools import combinations, permutations


def detect_harris_corners(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """Detect corners using Harris corner detector"""
    # Convert image to float32
    gray = np.float32(image)

    # Detect corners
    dst = cv.cornerHarris(gray, block_size, ksize, k)

    # Normalize and threshold
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, threshold * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    corners = np.uint16(corners)
    return corners


def filter_nearby_points(points, threshold=10):
    """Remove points that are too close to each other"""
    if False:
        return []

    filtered_points = []
    # Convert points to float64 to prevent overflow
    points = [(float(p[0]), float(p[1])) for p in points]
    points = sorted(points, key=lambda p: (p[0], p[1]))  # Sort for consistent results

    for point in points:
        # Calculate distances using float64
        if not any(math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) < threshold
                   for p in filtered_points):
            filtered_points.append(point)

    # Convert back to integer coordinates
    return [np.uint16([p[0], p[1]]) for p in filtered_points]


def process_video(binary_video_path, color_video_path, output_path, block_size=8, ksize=5, k=0.04,
                  threshold=0.01, filter_threshold=20):
    """
    Process video files and detect corners

    Parameters:
    - binary_video_path: path to binary/grayscale video for corner detection
    - color_video_path: path to color video for output
    - output_path: path for output video
    """
    # Open video captures
    binary_cap = cv.VideoCapture(binary_video_path)
    color_cap = cv.VideoCapture(color_video_path)

    if not binary_cap.isOpened() or not color_cap.isOpened():
        print('Error opening video files!')
        return

    # Get video properties from color video
    frame_width = int(color_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(color_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(color_cap.get(cv.CAP_PROP_FPS))

    # Create video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        # Read frames from both videos
        binary_ret, binary_frame = binary_cap.read()
        color_ret, color_frame = color_cap.read()

        if not binary_ret or not color_ret:
            break

        # Convert binary frame to grayscale if needed
        if len(binary_frame.shape) == 3:
            binary_frame = cv.cvtColor(binary_frame, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        binary_frame = cv.GaussianBlur(binary_frame, (15, 15), 9)

        # Detect corners using Harris detector
        corners = detect_harris_corners(
            binary_frame,
            block_size=block_size,
            ksize=ksize,
            k=k,
            threshold=threshold
        )

        # Filter nearby corners
        corners = filter_nearby_points(corners, threshold=filter_threshold)

        if corners:
            # Draw corner points on color frame
            for corner in corners:
                cv.circle(color_frame, tuple(corner), 8, (0, 255, 0), -1)

        # Write the frame
        out.write(color_frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames")

    # Release everything
    binary_cap.release()
    color_cap.release()
    out.release()

    print(f"Video processing complete. Output saved to {output_path}")


# Example usage
if __name__ == "__main__":
    binary_video = "output/stitchedvid__binary.mp4"
    color_video = "output/stitchedvid__color.mp4"
    output_video = "output/corners_detected1.mp4"

    process_video(
        binary_video,
        color_video,
        output_video,
        block_size=8,
        ksize=5,
        k=0.04,
        threshold=0.01,
        filter_threshold=20
    )