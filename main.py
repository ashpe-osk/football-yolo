import os
from utils.vid_utils import read_video, save_video
from track_player import Tracker

def main():
    # Video path
    video_path = r"C:\football-yolo\data\videos\vid001.mp4"
    model_path = r"C:\football-yolo\models\new_best.pt" 
    print(f"Reading video from: {video_path}")
    
    # Read the video frames
    video_frames = read_video(video_path)
    
    # Check if frames were loaded successfully
    if not video_frames:
        print("Failed to read video. Exiting...")
        return
    
    print(f"Successfully read {len(video_frames)} frames")
    
    # Output path
    output_path = "output_videos/combined_video.avi"
    print(f"Saving video to: {output_path}")

    # Initialize the Tracker
    tracker = Tracker(model_path=model_path)
    
    tracks = tracker.get_objects_tracks(
        video_frames,
        read_from_stub=False,  # CHANGE TO: False to regenerate tracks, True to read from stub
        stub_path="stubs/combined_track_stubs.pkl"
    )

    # Draw Output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save the video
    success = save_video(output_video_frames, output_path)
    
    if success:
        print("Video processing complete!")
    else:
        print("Failed to save video")

if __name__ == "__main__":
    main()