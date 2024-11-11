import cv2
import numpy as np
from pathlib import Path


def create_tiled_frame(original_frames, birds_eye, template_match, scale_factor=0.5):
    """
    Create a tiled visualization with cameras surrounding the birds-eye view.
    Layout:
           [North]
    [West][Birds Eye][East]     [Template Match]
           [South]
    """
    north, east, south, west = original_frames

    # Get dimensions and scale them down
    input_h, input_w = north.shape[:2]
    birdseye_h, birdseye_w = birds_eye.shape[:2]

    # Scale down the base dimensions
    birds_eye_target_h = int(birdseye_h * scale_factor)
    birds_eye_target_w = int(birdseye_w * scale_factor)

    # Size the N/S frames to match birds-eye width
    ns_target_w = birds_eye_target_w
    ns_target_h = int(input_h * (ns_target_w / input_w))

    # Size the E/W frames to match birds-eye height
    ew_target_h = birds_eye_target_h
    ew_target_w = int(input_w * (ew_target_h / input_h))

    # Resize all frames
    north_scaled = cv2.resize(north, (ns_target_w, ns_target_h))
    south_scaled = cv2.resize(south, (ns_target_w, ns_target_h))
    east_scaled = cv2.resize(east, (ew_target_w, ew_target_h))
    west_scaled = cv2.resize(west, (ew_target_w, ew_target_h))
    birds_eye_scaled = cv2.resize(birds_eye, (birds_eye_target_w, birds_eye_target_h))

    # Resize template matching visualization to match total height
    total_height = ns_target_h + birds_eye_target_h + ns_target_h
    template_width = int(template_match.shape[1] * total_height / template_match.shape[0])
    template_scaled = cv2.resize(template_match, (template_width, total_height))

    # Create the layout
    # First create the middle row (West + Birds Eye + East)
    middle_row = np.hstack([west_scaled, birds_eye_scaled, east_scaled])

    # Create empty space to match middle row width
    empty_space = np.zeros((ns_target_h, west_scaled.shape[1], 3), dtype=np.uint8)

    # Create top row (empty + North + empty)
    top_row = np.hstack([empty_space, north_scaled, empty_space])

    # Create bottom row (empty + South + empty)
    bottom_row = np.hstack([empty_space, south_scaled, empty_space])

    # Stack the rows
    main_view = np.vstack([top_row, middle_row, bottom_row])

    # Add compass direction labels with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6.0  # Increased font size
    font_thickness = 3
    font_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    padding = 20  # Padding for background rectangle

    # Calculate positions for labels
    h, w = main_view.shape[:2]
    label_positions = {
        'N': (w // 2, ns_target_h // 2),
        'S': (w // 2, h - ns_target_h // 2),
        'E': (w - ew_target_w // 2, h // 2),
        'W': (ew_target_w // 2, h // 2)
    }

    # Add labels with background
    for label, pos in label_positions.items():
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Calculate background rectangle position
        rect_x = pos[0] - text_w // 2 - padding // 2
        rect_y = pos[1] - text_h // 2 - padding // 2

        # Draw background rectangle
        cv2.rectangle(main_view,
                      (rect_x, rect_y),
                      (rect_x + text_w + padding, rect_y + text_h + padding),
                      bg_color, -1)

        # Draw text centered in the background
        text_x = pos[0] - text_w // 2
        text_y = pos[1] + text_h // 2
        cv2.putText(main_view, label, (text_x, text_y),
                    font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Combine with template matching view
    combined = np.hstack([main_view, template_scaled])

    # Add section titles with background
    title_font_scale = 6
    titles = [
        ("Camera Views & Birds Eye", (w // 3, 40)),
        ("Court Corner Detection", (w + template_scaled.shape[1] // 3, 40))
    ]

    for title, pos in titles:
        # Get text size for background
        (title_w, title_h), baseline = cv2.getTextSize(title, font, title_font_scale, font_thickness)

        # Draw background rectangle
        rect_x = pos[0] - title_w // 2 - padding
        rect_y = pos[1] - title_h - padding // 2
        cv2.rectangle(combined,
                      (rect_x, rect_y),
                      (rect_x + title_w + padding * 2, rect_y + title_h + padding),
                      bg_color, -1)

        # Draw text
        cv2.putText(combined, title,
                    (pos[0] - title_w // 2 + padding // 2, pos[1]+70),
                    font, title_font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Final size check - if still too large, scale down the entire frame
    max_dimension = 1920  # Maximum width allowed for MPEG-4
    if combined.shape[1] > max_dimension:
        scale = max_dimension / combined.shape[1]
        final_size = (max_dimension, int(combined.shape[0] * scale))
        combined = cv2.resize(combined, final_size)

    return combined


def create_visualization(north_path, east_path, south_path, west_path,
                         birds_eye_path, template_match_path, output_path,
                         scale_factor=0.5):
    """
    Create a combined visualization video from the input videos.
    """
    # Verify all input files exist
    input_paths = [north_path, east_path, south_path, west_path,
                   birds_eye_path, template_match_path]

    for path in input_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    # Open all input videos
    caps = []
    try:
        caps = [cv2.VideoCapture(str(p)) for p in input_paths]

        # Verify all videos opened successfully
        for i, cap in enumerate(caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {input_paths[i]}")

        # Get video properties from first video
        fps = caps[0].get(cv2.CAP_PROP_FPS)

        # Create output video writer after we get first frame and know dimensions
        out = None
        frame_count = 0

        # Create display window
        cv2.namedWindow('Combined View', cv2.WINDOW_NORMAL)

        print("Creating visualization...")
        while True:
            # Read frames from all videos
            frames = []
            all_read_successful = True

            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    print(f"\nReached end of video: {input_paths[i]}")
                    all_read_successful = False
                    break
                frames.append(frame)

            if not all_read_successful:
                break

            # Split frames into their components
            north, east, south, west = frames[:4]
            birds_eye = frames[4]
            template_match = frames[5]

            # Create combined visualization
            combined_frame = create_tiled_frame(
                [north, east, south, west],
                birds_eye,
                template_match,
                scale_factor
            )

            # Initialize video writer if not done yet
            if out is None:
                h, w = combined_frame.shape[:2]
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))
                if not out.isOpened():
                    raise RuntimeError(f"Failed to create output video: {output_path}")
                print(f"Output video dimensions: {w}x{h}")

            # Write frame
            out.write(combined_frame)

            # Display frame
            cv2.imshow('Combined View', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing stopped by user")
                break

            frame_count += 1
            print(f'\rProcessed frames: {frame_count}', end='')

        if frame_count > 0:
            print(f"\nVisualization complete! Saved to {output_path}")
            print(f"Total frames processed: {frame_count}")
        else:
            print("\nNo frames were processed!")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

    finally:
        # Clean up
        for cap in caps:
            if cap is not None:
                cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example paths - replace these with your actual video paths
    north_path = "vid_north.mkv"
    east_path = "vid_east.mkv"
    south_path = "vid_south.mkv"
    west_path = "vid_west.mkv"
    birds_eye_path = "output/stitchedvid__color.mp4"
    template_match_path = "output/corners_detected1.mp4"
    output_path = "combined_visualization1.mp4"

    try:
        create_visualization(
            north_path, east_path, south_path, west_path,
            birds_eye_path, template_match_path, output_path, scale_factor=0.5
        )
    except Exception as e:
        print(f"Failed to create visualization: {str(e)}")