import cv2
import os

def read_video(video_path):
    """
    Read video frames from a file
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return []
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            print(f"Loaded {frame_count} frames...")
    
    # Release the video capture
    cap.release()
    
    # Check if any frames were read
    if len(frames) == 0:
        print(f"Error: No frames could be read from {video_path}")
        return []
    
    print(f"Successfully loaded {len(frames)} frames")
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Save frames as a video file
    """
    # Check if there are frames to save
    if not output_video_frames:
        print("Error: No frames to save")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Get frame dimensions from the first frame
        height, width = output_video_frames[0].shape[:2]
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
        
        # Check if VideoWriter was created successfully
        if not out.isOpened():
            print(f"Error: Could not create video writer for {output_video_path}")
            return False
        
        # Write all frames
        for i, frame in enumerate(output_video_frames):
            out.write(frame)
            # Progress update every 100 frames
            if (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(output_video_frames)} frames...")
        
        # Release the video writer
        out.release()
        
        print(f"Video successfully saved to: {output_video_path}")
        print(f"Total frames saved: {len(output_video_frames)}")
        return True
        
    except Exception as e:
        print(f"Error saving video: {e}")
        return False