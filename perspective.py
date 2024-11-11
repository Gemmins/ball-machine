import numpy as np
import cv2
import os
from typing import Union, Tuple, List


def calculate_perspective_points(image_height=1080, image_width=1920, angle_degrees=30, vertical_crop_factor=2 / 3):
    """Calculate source and destination points for perspective transform."""
    ground_angle_rad = np.deg2rad(90 - angle_degrees)
    expansion = 1 / np.cos(ground_angle_rad)

    center_x = image_width / 2
    bottom_offset = image_height * (1 - vertical_crop_factor)

    src_top_width = image_width
    src_bottom_width = image_width * expansion

    src_points = np.float32([
        [center_x - src_top_width / 2, bottom_offset],
        [center_x + src_top_width / 2, bottom_offset],
        [center_x + src_bottom_width / 2, image_height],
        [center_x - src_bottom_width / 2, image_height]
    ])

    dst_points = np.float32([
        [center_x - src_top_width / 2, bottom_offset],
        [center_x + src_top_width / 2, bottom_offset],
        [center_x + src_top_width / 2, image_height],
        [center_x - src_top_width / 2, image_height]
    ])

    return src_points, dst_points


def apply_directional_transform(image, angle_degrees=30):
    """Apply perspective transform to an image."""
    height, width = image.shape[:2]
    src_points, dst_points = calculate_perspective_points(height, width, angle_degrees)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, matrix, (width, height))
    return transformed


def transform_and_rotate(image, angle_degrees=30, rotation=None, h_scale=1.0, v_scale=1.0):
    """Transform image, rotate if needed, and apply scaling in both directions."""
    transformed = apply_directional_transform(image, angle_degrees)

    # Scale the transformed image before rotation
    if rotation in ['clockwise', 'counterclockwise']:
        # For east/west images, apply vertical scaling
        new_height = int(transformed.shape[0] * v_scale)
        transformed = cv2.resize(transformed, (transformed.shape[1], new_height))
    else:
        # For north/south images, apply horizontal scaling
        new_width = int(transformed.shape[0] * h_scale)
        transformed = cv2.resize(transformed, (transformed.shape[1], new_width))

    if rotation == 'clockwise':
        return cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 'counterclockwise':
        return cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == '180':
        return cv2.rotate(transformed, cv2.ROTATE_180)
    return transformed


def add_to_region(output, img, y_start, y_end, x_start, x_end):
    """Add image to region of output array, only adding where output is zero."""
    y_start = max(0, y_start)
    y_end = min(output.shape[0], y_end)
    x_start = max(0, x_start)
    x_end = min(output.shape[1], x_end)

    region_h = y_end - y_start
    region_w = x_end - x_start

    if region_h <= 0 or region_w <= 0:
        return

    out_region = output[y_start:y_end, x_start:x_end]
    img_region = img[:region_h, :region_w]

    mask = (img_region != 0).any(axis=2)
    out_region[mask] = img_region[mask]
    output[y_start:y_end, x_start:x_end] = out_region


def stitch_frame(north_img, east_img, south_img, west_img, angle_degrees=30,
                 h_scale=0.85, v_scale=0.85, camera_radius_pixels=110, threshold=128):
    """Transform and stitch images/frames together, returning both color and binary versions."""
    h, w = north_img.shape[:2]
    output = np.zeros((h * 2 + camera_radius_pixels * 2, w * 2 + camera_radius_pixels * 2, 3), dtype=np.uint8)

    center_h = output.shape[0] // 2
    center_w = output.shape[1] // 2

    # Transform all images
    north_transformed = transform_and_rotate(north_img, angle_degrees, None, h_scale, v_scale)
    east_transformed = transform_and_rotate(east_img, angle_degrees, 'clockwise', h_scale, v_scale)
    south_transformed = transform_and_rotate(south_img, angle_degrees, '180', h_scale, v_scale)
    west_transformed = transform_and_rotate(west_img, angle_degrees, 'counterclockwise', h_scale, v_scale)

    # Place north image (top)
    n_h, n_w = north_transformed.shape[:2]
    add_to_region(output, north_transformed,
                  center_h - n_h - camera_radius_pixels,
                  center_h - camera_radius_pixels,
                  center_w - n_w // 2,
                  center_w + n_w // 2)

    # Place south image (bottom)
    s_h, s_w = south_transformed.shape[:2]
    add_to_region(output, south_transformed,
                  center_h + camera_radius_pixels,
                  center_h + s_h + camera_radius_pixels,
                  center_w - s_w // 2,
                  center_w + s_w // 2)

    # Place east image (right)
    e_h, e_w = east_transformed.shape[:2]
    add_to_region(output, east_transformed,
                  center_h - e_h // 2,
                  center_h + e_h // 2,
                  center_w + camera_radius_pixels,
                  center_w + e_w + camera_radius_pixels)

    # Place west image (left)
    w_h, w_w = west_transformed.shape[:2]
    add_to_region(output, west_transformed,
                  center_h - w_h // 2,
                  center_h + w_h // 2,
                  center_w - w_w - camera_radius_pixels,
                  center_w - camera_radius_pixels)

    # Create binary version
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return output, binary_color


def load_input(source: str) -> Union[np.ndarray, cv2.VideoCapture]:
    """Load either an image or video file."""
    if source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return cv2.VideoCapture(source)
    else:
        return cv2.imread(source)


def process_inputs(north_source: str, east_source: str, south_source: str, west_source: str,
                   output_dir: str, angle_degrees=9, h_scale=0.87, v_scale=0.87,
                   camera_radius_pixels=99, threshold=130):
    """Process either images or videos and save both color and binary results."""

    # Load all inputs
    inputs = [load_input(src) for src in [north_source, east_source, south_source, west_source]]

    # Check if we're dealing with videos
    is_video = all(isinstance(inp, cv2.VideoCapture) for inp in inputs)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get base name from north source for output naming
    base_name = os.path.splitext(os.path.basename(north_source))[0].replace('north', '')

    if is_video:
        # Get video properties
        fps = inputs[0].get(cv2.CAP_PROP_FPS)
        frame_width = int(inputs[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(inputs[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create sample frame to get dimensions
        sample_color, sample_binary = stitch_frame(
            *[np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in range(4)],
            angle_degrees, h_scale, v_scale, camera_radius_pixels, threshold
        )

        # Lists to store frames
        color_frames = []
        binary_frames = []

        # Create windows for display
        cv2.namedWindow('Color Output', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Binary Output', cv2.WINDOW_NORMAL)

        frame_count = 0

        print("Processing frames...")
        while True:
            # Read frames from all videos
            frames = []
            for cap in inputs:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) != 4:
                break

            # Process frames
            color_result, binary_result = stitch_frame(*frames, angle_degrees, h_scale, v_scale,
                                                       camera_radius_pixels, threshold)

            # Store frames
            color_frames.append(color_result)
            binary_frames.append(binary_result)

            # Display frames
            # Scale down for display if too large
            scale = min(1440 / color_result.shape[1], 900 / color_result.shape[0])
            if scale < 1:
                display_width = int(color_result.shape[1] * scale)
                display_height = int(color_result.shape[0] * scale)
                color_display = cv2.resize(color_result, (display_width, display_height))
                binary_display = cv2.resize(binary_result, (display_width, display_height))
            else:
                color_display = color_result
                binary_display = binary_result

            cv2.imshow('Color Output', color_display)
            cv2.imshow('Binary Output', binary_display)

            # Update frame count and display
            frame_count += 1
            print(f'\rProcessing frame: {frame_count}', end='')

            # Check for 'q' key to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f'\nProcessed {frame_count} frames')
        print("Writing video files...")

        # Create video writers and write all frames
        if color_frames:  # Only if we have frames to write
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            color_path = os.path.join(output_dir, f'stitched{base_name}_color.mp4')
            binary_path = os.path.join(output_dir, f'stitched{base_name}_binary.mp4')

            # Get dimensions from the first frame
            height, width = color_frames[0].shape[:2]

            # Create video writers
            out_color = cv2.VideoWriter(color_path, fourcc, fps, (width, height))
            out_binary = cv2.VideoWriter(binary_path, fourcc, fps, (width, height))

            # Write all frames at once
            for color_frame, binary_frame in zip(color_frames, binary_frames):
                out_color.write(color_frame)
                out_binary.write(binary_frame)

            # Clean up
            out_color.release()
            out_binary.release()

        # Clean up video captures
        for cap in inputs:
            cap.release()
        cv2.destroyAllWindows()

        print("Done!")

    else:
        # Process single images
        color_result, binary_result = stitch_frame(*inputs, angle_degrees, h_scale, v_scale,
                                                   camera_radius_pixels, threshold)

        # Save images
        cv2.imwrite(os.path.join(output_dir, f'stitched{base_name}_color.png'), color_result)
        cv2.imwrite(os.path.join(output_dir, f'stitched{base_name}_binary.png'), binary_result)

        # Display images
        cv2.namedWindow('Color Output', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Binary Output', cv2.WINDOW_NORMAL)

        # Scale down for display if too large
        scale = min(1440 / color_result.shape[1], 900 / color_result.shape[0])
        if scale < 1:
            display_width = int(color_result.shape[1] * scale)
            display_height = int(color_result.shape[0] * scale)
            color_display = cv2.resize(color_result, (display_width, display_height))
            binary_display = cv2.resize(binary_result, (display_width, display_height))
        else:
            color_display = color_result
            binary_display = binary_result

        cv2.imshow('Color Output', color_display)
        cv2.imshow('Binary Output', binary_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage - just modify these source paths
north_source = 'output_quarters/cnorth.jpg'  # or .mp4
east_source = 'output_quarters/ceast.jpg'  # or .mp4
south_source = 'output_quarters/csouth.jpg'  # or .mp4
west_source = 'output_quarters/cwest.jpg'  # or .mp4
output_dir = 'output'

# Process the inputs
process_inputs(
    north_source,
    east_source,
    south_source,
    west_source,
    output_dir,
    angle_degrees=11,
    h_scale=0.87,
    v_scale=0.87,
    camera_radius_pixels=95,
    threshold=130)
