import copy
import pickle
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import Mock

from track_player.track import Tracker, TrackerConfig
from team_allocator import TeamAllocator
from utils.vid_utils import read_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_STUB = PROJECT_ROOT / "stubs" / "combined_track_stubs.pkl"
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "vid_for_data_30s.mp4"


class TrackerReplayTests(unittest.TestCase):
    @unittest.skipUnless(
        TRACK_STUB.exists() and VIDEO_PATH.exists(),
        "Saved replay stub and source video are required."
    )
    def test_cached_replay_preserves_global_id_invariants(self):
        with TRACK_STUB.open("rb") as stub_file:
            original_tracks = pickle.load(stub_file)
        tracks = copy.deepcopy(original_tracks)
        frames = read_video(str(VIDEO_PATH))
        self.assertEqual(len(tracks["players"]), len(frames))

        tracker = Tracker.__new__(Tracker)
        tracker.config = TrackerConfig(debug_invariants=True)
        tracker.global_max_age = tracker.config.global_max_age
        tracker.gmc_max_corners = 300
        tracker.gmc_min_points = 12
        tracker.pitch_status_per_frame = (
            original_tracks.get("_metadata", {}).get(
                "polygon_status_per_frame",
                ["cached"] * len(frames)
            )
        )

        replayed = tracker._associate_global_ids(frames, tracks)

        self.assertEqual(len(replayed["players"]), len(frames))
        self.assertLessEqual(
            len(tracker.global_tracks),
            tracker.config.max_global_ids
        )
        self.assertLessEqual(
            len(tracker.tid_to_gid),
            tracker.config.max_global_ids
        )
        self.assertTrue(all(
            item["active"] <= tracker.config.max_global_ids
            for item in tracker.association_diagnostics
        ))
        self.assertEqual(
            len(tracker.tid_to_gid),
            len(tracker.gid_to_tid)
        )
        for tid, gid in tracker.tid_to_gid.items():
            self.assertEqual(tracker.gid_to_tid[gid], tid)

        for frame_tracks in replayed["players"]:
            global_ids = [
                player["global_id"] for player in frame_tracks.values()
            ]
            self.assertEqual(len(global_ids), len(set(global_ids)))
            self.assertEqual(
                set(frame_tracks),
                set(global_ids)
            )

        with TRACK_STUB.open("rb") as stub_file:
            unchanged_tracks = pickle.load(stub_file)
        self.assertEqual(original_tracks, unchanged_tracks)

    def test_mapping_expiry_revalidates_reappearing_tracker_id(self):
        config = replace(
            TrackerConfig(debug_invariants=True),
            tid_mapping_max_age=3,
            min_confirm_frames=2,
            tentative_max_gap=1,
            global_max_age=50
        )
        tracker = Tracker.__new__(Tracker)
        tracker.config = config
        tracker.global_max_age = config.global_max_age
        tracker.gmc_max_corners = config.camera_maximum_feature_points
        tracker.gmc_min_points = config.camera_minimum_feature_points
        frames = [np.zeros((90, 160, 3), dtype=np.uint8) for _ in range(40)]
        player_frames = []
        for frame_num in range(len(frames)):
            if frame_num < 2 or frame_num >= 36:
                player_frames.append({
                    7: {
                        "tracker_id": 7,
                        "bbox": [60.0, 20.0, 80.0, 70.0]
                    }
                })
            else:
                player_frames.append({})

        replayed = tracker._associate_global_ids(
            frames,
            {"players": player_frames}
        )

        self.assertTrue(any(
            (7, 1) in item["mapping_expired"]
            for item in tracker.association_diagnostics
        ))
        self.assertEqual(tracker.tid_to_gid[7], 1)
        self.assertEqual(tracker.gid_to_tid[1], 7)
        self.assertTrue(replayed["players"][-1])

    def test_roster_full_detection_is_kept_for_neutral_ellipse(self):
        config = replace(
            TrackerConfig(debug_invariants=True),
            max_global_ids=1,
            min_confirm_frames=1
        )
        tracker = Tracker.__new__(Tracker)
        tracker.config = config
        tracker.global_max_age = config.global_max_age
        tracker.gmc_max_corners = config.camera_maximum_feature_points
        tracker.gmc_min_points = config.camera_minimum_feature_points
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        tracker.draw_team_ball_control = Mock(
            side_effect=lambda image, *_args, **_kwargs: image
        )
        tracks = {
            "ball": [{}],
            "referees": [{}],
            "players": [{
                1: {
                    "tracker_id": 1,
                    "bbox": [10.0, 10.0, 30.0, 70.0]
                },
                2: {
                    "tracker_id": 2,
                    "bbox": [110.0, 10.0, 130.0, 70.0]
                }
            }],
            "_metadata": {}
        }

        replayed = tracker._associate_global_ids([frame], tracks)

        self.assertEqual(len(replayed["players"][0]), 1)
        unassigned = replayed["_metadata"]["unassigned_players_per_frame"][0]
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(unassigned[0]["tracker_id"], 2)
        annotated = tracker.draw_annotations(
            [frame],
            replayed,
            np.array([0])
        )[0]
        neutral = np.asarray(
            tracker.UNASSIGNED_PLAYER_COLOR,
            dtype=np.uint8
        )
        self.assertTrue(np.any(np.all(annotated == neutral, axis=2)))

    def test_global_ids_follow_players_when_tracker_ids_swap(self):
        config = replace(
            TrackerConfig(debug_invariants=True),
            max_global_ids=2,
            min_confirm_frames=2,
            global_max_age=20,
            forced_position_base_px=20,
            forced_max_speed_px_per_frame=5,
            position_gate_base_px=20,
            position_gate_age_speed_px_per_frame=5
        )
        tracker = Tracker.__new__(Tracker)
        tracker.config = config
        tracker.global_max_age = config.global_max_age
        tracker.gmc_max_corners = config.camera_maximum_feature_points
        tracker.gmc_min_points = config.camera_minimum_feature_points
        frames = []
        player_frames = []
        for frame_num in range(5):
            frame = np.full((120, 240, 3), (35, 130, 35), dtype=np.uint8)
            frame[30:100, 30:60] = (180, 40, 20)
            frame[30:100, 170:200] = (220, 220, 220)
            frames.append(frame)
            left_tracker_id, right_tracker_id = (
                (1, 2) if frame_num < 3 else (2, 1)
            )
            player_frames.append({
                left_tracker_id: {
                    "tracker_id": left_tracker_id,
                    "bbox": [30.0, 30.0, 60.0, 100.0]
                },
                right_tracker_id: {
                    "tracker_id": right_tracker_id,
                    "bbox": [170.0, 30.0, 200.0, 100.0]
                }
            })

        replayed = tracker._associate_global_ids(
            frames,
            {"players": player_frames}
        )
        left_global_ids = [
            gid
            for gid, player in replayed["players"][1].items()
            if player["bbox"][0] < 100
        ]
        right_global_ids = [
            gid
            for gid, player in replayed["players"][1].items()
            if player["bbox"][0] > 100
        ]
        self.assertEqual(len(left_global_ids), 1)
        self.assertEqual(len(right_global_ids), 1)
        left_gid = left_global_ids[0]
        right_gid = right_global_ids[0]

        for frame_num in (3, 4):
            players = replayed["players"][frame_num]
            self.assertTrue(any(
                gid == left_gid and player["bbox"][0] < 100
                for gid, player in players.items()
            ))
            self.assertTrue(any(
                gid == right_gid and player["bbox"][0] > 100
                for gid, player in players.items()
            ))

    def test_expired_global_id_is_not_recycled_for_another_player(self):
        config = replace(
            TrackerConfig(debug_invariants=True),
            max_global_ids=1,
            global_max_age=2,
            tid_mapping_max_age=2,
            min_confirm_frames=2,
            tentative_max_gap=1
        )
        tracker = Tracker.__new__(Tracker)
        tracker.config = config
        tracker.global_max_age = config.global_max_age
        tracker.gmc_max_corners = config.camera_maximum_feature_points
        tracker.gmc_min_points = config.camera_minimum_feature_points
        frames = [np.zeros((90, 160, 3), dtype=np.uint8) for _ in range(7)]
        player_frames = [
            {1: {"tracker_id": 1, "bbox": [20.0, 20.0, 40.0, 70.0]}},
            {1: {"tracker_id": 1, "bbox": [20.0, 20.0, 40.0, 70.0]}},
            {},
            {},
            {},
            {2: {"tracker_id": 2, "bbox": [100.0, 20.0, 120.0, 70.0]}},
            {2: {"tracker_id": 2, "bbox": [100.0, 20.0, 120.0, 70.0]}}
        ]

        replayed = tracker._associate_global_ids(
            frames,
            {"players": player_frames}
        )

        self.assertEqual(set(tracker.global_tracks), {1})
        self.assertEqual(tracker.global_tracks[1]["last_tracker_id"], 1)
        self.assertTrue(all(
            not frame_tracks for frame_tracks in replayed["players"][5:]
        ))
        self.assertTrue(any(
            frame_tracks for frame_tracks in
            replayed["_metadata"]["unassigned_players_per_frame"][5:]
        ))

    def test_team_assignment_uses_aggregated_track_color_samples(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[:] = (40, 120, 40)
        boxes = {
            1: {"bbox": [10, 10, 50, 110]},
            2: {"bbox": [60, 10, 100, 110]},
            3: {"bbox": [110, 10, 150, 110]}
        }
        colors = {
            1: (45, 65, 115),
            2: (55, 75, 125),
            3: (205, 215, 225)
        }
        for player_id, player in boxes.items():
            x1, y1, x2, y2 = player["bbox"]
            frame[y1:y2, x1:x2] = colors[player_id]

        allocator = TeamAllocator()
        allocator.allocate_teams(frame, boxes)
        for player_id, player in boxes.items():
            allocator.get_player_team(frame, player["bbox"], player_id)

        brighter_frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=25)
        for player_id, player in boxes.items():
            allocator.get_player_team(
                brighter_frame,
                player["bbox"],
                player_id
            )

        stable_teams = allocator.finalize_player_teams()
        self.assertEqual(stable_teams[1], stable_teams[2])
        self.assertNotEqual(stable_teams[1], stable_teams[3])


if __name__ == "__main__":
    unittest.main()
