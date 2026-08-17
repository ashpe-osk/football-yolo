import os

import cv2
from utils.vid_utils import read_video, save_video
from track_player import Tracker
from team_allocator import TeamAllocator


def main():
    # ---------------------------------------------------------
    # VIDEO / MODEL PATHS
    # ---------------------------------------------------------

    video_path = r"C:\football-yolo\data\videos\vid001.mp4"
    model_path = r"C:\football-yolo\models\new_best.pt"

    print(f"Reading video from: {video_path}")

    # ---------------------------------------------------------
    # READ VIDEO
    # ---------------------------------------------------------

    video_frames = read_video(video_path)

    if not video_frames:
        print("Failed to read video. Exiting...")
        return

    print(f"Successfully read {len(video_frames)} frames")

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    output_path = "output_videos/combined_video.avi"

    print(f"Saving video to: {output_path}")

    # ---------------------------------------------------------
    # INITIALIZE TRACKER
    # ---------------------------------------------------------

    tracker = Tracker(model_path=model_path)

    # ---------------------------------------------------------
    # GET TRACKS
    # ---------------------------------------------------------

    tracks = tracker.get_objects_tracks(
        video_frames,
        read_from_stub=False,
        stub_path="stubs/combined_track_stubs.pkl"
    )

    # ---------------------------------------------------------
    # INITIALIZE TEAM ALLOCATOR
    # ---------------------------------------------------------

    team_allocator = TeamAllocator()

    # ---------------------------------------------------------
    # FIND A GOOD INITIAL FRAME
    # ---------------------------------------------------------
    #
    # Instead of blindly using frame 0, look through the first
    # 50 frames and select the frame containing the most players.
    #
    # This gives KMeans more useful player observations for
    # establishing the two team colors.
    #

    best_frame_num = None
    max_players = 0

    search_frames = min(50, len(video_frames))

    for frame_num in range(search_frames):

        player_tracks = tracks["players"][frame_num]

        player_count = len(player_tracks)

        if player_count > max_players:
            max_players = player_count
            best_frame_num = frame_num

    if best_frame_num is None or max_players < 2:
        print("Could not find enough players to initialize teams.")
        return

    print(
        f"Using frame {best_frame_num} for team initialization "
        f"({max_players} players detected)"
    )

    # ---------------------------------------------------------
    # INITIALIZE TEAM COLORS
    # ---------------------------------------------------------

    team_allocator.allocate_teams(
        video_frames[best_frame_num],
        tracks["players"][best_frame_num]
    )

    # ---------------------------------------------------------
    # ASSIGN TEAMS TO ALL PLAYERS
    # ---------------------------------------------------------

    for frame_num, player_track in enumerate(
        tracks["players"]
    ):

        frame = video_frames[frame_num]

        for player_id, track in player_track.items():

            bbox = track["bbox"]

            # player_id is now the GLOBAL ID
            team = team_allocator.get_player_team(
                frame,
                bbox,
                player_id
            )

            # Save team assignment
            tracks["players"][frame_num][player_id]["team"] = team

            # Save team color
            tracks["players"][frame_num][player_id][
                "team_color"
            ] = team_allocator.team_colors[team]

    # ---------------------------------------------------------
    # DRAW OUTPUT
    # ---------------------------------------------------------

    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks
    )

    # ---------------------------------------------------------
    # SAVE VIDEO
    # ---------------------------------------------------------

    success = save_video(
        output_video_frames,
        output_path
    )

    if success:
        print("Video processing complete!")
    else:
        print("Failed to save video")


if __name__ == "__main__":
    main()