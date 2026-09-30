import os

import numpy as np
from utils.vid_utils import get_video_fps, read_video, save_video
from track_player import Tracker
from team_allocator import TeamAllocator
from player_ball_assigner import PlayerBallAssigner
from passing_stats.passing_stats import compute_pass_stats, save_pass_stats


# OpenCV colors are BGR. These are annotation colors, not kit-color
# predictions; team membership still comes from the jersey-color clusters.
DISPLAY_TEAM_COLORS = {
    1: (0, 0, 255),
    2: (0, 255, 255),
}


def main():
    video_path = r"C:\football-yolo\data\videos\vid_for_data_30s.mp4"
    model_path = r"C:\football-yolo\models\new_best.pt"

    print(f"Reading video from: {video_path}")

    video_frames = read_video(video_path)

    if not video_frames:
        print("Failed to read video. Exiting...")
        return

    source_fps = get_video_fps(video_path)
    print(f"Successfully read {len(video_frames)} frames")
    print(f"Source video frame rate: {source_fps:.3f} fps")

    output_path = "output_videos/combined_video.avi"

    print(f"Saving video to: {output_path}")

    tracker = Tracker(model_path=model_path)

    tracks = tracker.get_objects_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/combined_track_stubs.pkl",
        allow_recompute=False
    )
    tracks = tracker.remove_players_overlapping_referees(tracks)
    unassigned_frames = tracks.get("_metadata", {}).get(
        "unassigned_players_per_frame",
        []
    )
    unassigned_count = sum(len(frame_tracks) for frame_tracks in unassigned_frames)
    if unassigned_count:
        print(
            f"Rendering {unassigned_count} detections without a stable global "
            "ID as neutral ellipses.",
            flush=True
        )

    if (
        len(tracker.camera_transform_per_frame) != len(video_frames)
        or len(tracker.camera_movement_per_frame) != len(video_frames)
    ):
        print("Computing camera transforms for motion-compensated ball tracking...", flush=True)
        tracker.compute_camera_movement(video_frames)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_allocator = TeamAllocator()

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

    team_allocator.allocate_teams(
        video_frames[best_frame_num],
        tracks["players"][best_frame_num]
    )
    team_labels = {1: "Team 1", 2: "Team 2"}

    last_team_sample_frame = {}
    team_sample_interval = 10
    for frame_num, player_track in enumerate(tracks["players"]):
        if frame_num % 100 == 0:
            print(
                f"Collecting team-color samples: {frame_num + 1}/"
                f"{len(tracks['players'])} frames",
                flush=True
            )
        frame = video_frames[frame_num]
        for player_id, track in player_track.items():
            last_sample_frame = last_team_sample_frame.get(player_id)
            if (
                last_sample_frame is not None
                and frame_num - last_sample_frame < team_sample_interval
            ):
                continue
            bbox = track["bbox"]
            team_allocator.get_player_team(frame, bbox, player_id)
            last_team_sample_frame[player_id] = frame_num

    stable_team_by_player = team_allocator.finalize_player_teams()
    team_counts = {
        team: sum(assigned_team == team for assigned_team in stable_team_by_player.values())
        for team in (1, 2)
    }
    print(f"Stable team-color assignments: {team_counts}", flush=True)
    for player_track in tracks["players"]:
        for player_id, track in player_track.items():
            team = stable_team_by_player.get(player_id)
            if team is not None:
                track["team"] = team
                track["team_color"] = DISPLAY_TEAM_COLORS[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        for player in player_track.values():
            player["has_ball"] = False

        ball_track = tracks["ball"][frame_num].get(1)
        if ball_track is None:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue
        assigned_player = player_assigner.assign_ball_to_players(
            player_track,
            ball_track["bbox"]
        )
        if assigned_player != -1:
            assigned_track = tracks["players"][frame_num][assigned_player]
            if assigned_track.get("team") in (1, 2):
                assigned_track["has_ball"] = True
                team_ball_control.append(assigned_track["team"])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    pass_stats = compute_pass_stats(
        tracks,
        fps=source_fps,
        camera_movement=tracker.camera_movement_per_frame,
        team_labels=team_labels
    )
    pass_stats_path = os.path.join(
        os.path.dirname(output_path),
        "team_pass_stats.json"
    )
    save_pass_stats(pass_stats, pass_stats_path)

    tracks = tracker.add_positions_to_tracks(tracks)

    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control,
        team_labels,
        DISPLAY_TEAM_COLORS,
        pass_stats["timeline"]
    )

    output_video_frames = tracker.draw_camera_movement(
        output_video_frames,
        tracker.camera_movement_per_frame
    )

    success = save_video(output_video_frames, output_path, fps=source_fps)
    if success:
        print("Video processing complete!")
    else:
        print("Failed to save video")


if __name__ == "__main__":
    main()