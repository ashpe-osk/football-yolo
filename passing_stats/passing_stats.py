"""
Team-level pass statistics that do NOT depend on stable player IDs.

A pass is detected from ball possession, not player identity:
  1. Each frame, the controlling team = team of the player with has_ball.
  2. Control must last `min_control_frames` frames to be confirmed (removes flicker).
  3. Between two consecutive confirmed control segments:
       - gap too long                -> skipped (ball lost / out of play / cut)
       - ball travelled < min dist   -> skipped (dribble, tackle, flicker)
       - same team receives          -> COMPLETE pass for that team
       - other team receives         -> INCOMPLETE pass (intercepted)
Events deliberately contain team-level information only. Player IDs are
excluded until the tracking IDs are stable enough to support attribution.

Expected tracks format (what your Tracker + team assigner already produce):
  tracks["players"][frame][global_id] = {"bbox":..., "team": 1|2, "has_ball": bool, ...}
  tracks["ball"][frame][1] = {"bbox": [...]}
"""
import json
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class PassConfig:
    team_key: str = "team"                 # key holding the team id on each player
    has_ball_key: str = "has_ball"         # key holding the possession flag
    min_control_frames: int = 3            # frames of steady control to confirm possession
    max_flight_frames: int = 45            # max frames the ball may be in flight (~1.8 s @ 25 fps)
    min_pass_distance_px: float = 60.0     # min ball travel to call it a pass
    compensate_camera: bool = True         # subtract cumulative camera motion from positions


def _camera_offsets(n_frames, camera_movement, cfg):
    if cfg.compensate_camera:
        if camera_movement is not None and len(camera_movement) == n_frames:
            return np.cumsum(np.asarray(camera_movement, dtype=np.float32), axis=0)
        print("WARNING: camera_movement missing or wrong length; "
              "distances are not camera-compensated.")
    return np.zeros((n_frames, 2), dtype=np.float32)


def _frame_control(players, cfg):
    """Team currently controlling the ball in one frame (None if nobody)."""
    teams = [
        p.get(cfg.team_key)
        for p in players.values()
        if p.get(cfg.has_ball_key) and p.get(cfg.team_key) is not None
    ]
    if not teams:
        return None
    return Counter(teams).most_common(1)[0][0]


def _ball_at(tracks, cam, frames):
    """Mean camera-compensated ball centre over the given frames (None if unseen)."""
    points = []
    for f in frames:
        ball = tracks["ball"][f].get(1)
        if ball:
            x1, y1, x2, y2 = ball["bbox"]
            points.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32) - cam[f])
    return np.mean(points, axis=0) if points else None


def compute_pass_stats(tracks, fps=25.0, camera_movement=None, cfg=None, team_labels=None):
    cfg = cfg or PassConfig()
    n = len(tracks["players"])
    cam = _camera_offsets(n, camera_movement, cfg)

    # 1) per-frame controlling team
    controls = [_frame_control(tracks["players"][f], cfg) for f in range(n)]

    # 2) confirmed control segments (run-length encode, drop short runs)
    segments = []
    f = 0
    while f < n:
        team, g = controls[f], f
        while g + 1 < n and controls[g + 1] == team:
            g += 1
        if team is not None and (g - f + 1) >= cfg.min_control_frames:
            segments.append({"team": team, "start": f, "end": g})
        f = g + 1

    # 3) classify transitions between consecutive segments
    events = []
    skipped = Counter()
    for a, b in zip(segments, segments[1:]):
        gap = b["start"] - a["end"] - 1
        if gap > cfg.max_flight_frames:
            skipped["gap_too_long"] += 1
            continue

        exit_pos = _ball_at(tracks, cam, range(max(a["end"] - 2, a["start"]), a["end"] + 1))
        entry_pos = _ball_at(tracks, cam, range(b["start"], min(b["start"] + 3, b["end"] + 1)))
        if exit_pos is None or entry_pos is None:
            skipped["ball_not_visible"] += 1
            continue

        distance = float(np.linalg.norm(entry_pos - exit_pos))
        if distance < cfg.min_pass_distance_px:
            skipped["too_short"] += 1
            continue

        complete = a["team"] == b["team"]
        events.append({
            "frame": b["start"],
            "time_s": round(b["start"] / fps, 2),
            "team": a["team"],
            "outcome": "complete" if complete else "incomplete",
            "receiver_team": b["team"],
            "release_frame": a["end"],
            "receive_frame": b["start"],
            "flight_frames": gap,
            "distance_px": round(distance, 1),
        })

    # 4) aggregate per team
    team_ids = sorted(
        {1, 2}
        | {s["team"] for s in segments}
        | {e["team"] for e in events}
    )
    teams = {}
    for t in team_ids:
        complete = sum(1 for e in events if e["team"] == t and e["outcome"] == "complete")
        incomplete = sum(1 for e in events if e["team"] == t and e["outcome"] == "incomplete")
        attempted = complete + incomplete
        teams[str(t)] = {
            "label": (team_labels or {}).get(t, f"Team {t}"),
            "passes_attempted": attempted,
            "passes_completed": complete,
            "passes_incomplete": incomplete,
            "pass_completion_pct": round(100.0 * complete / attempted, 1) if attempted else None,
        }

    events_by_frame = {}
    for event in events:
        events_by_frame.setdefault(event["frame"], []).append(event)

    running = {
        team_id: {
            "passes_attempted": 0,
            "passes_completed": 0,
            "passes_incomplete": 0,
        }
        for team_id in team_ids
    }
    timeline = []
    for frame_num in range(n):
        for event in events_by_frame.get(frame_num, []):
            counters = running[event["team"]]
            counters["passes_attempted"] += 1
            if event["outcome"] == "complete":
                counters["passes_completed"] += 1
            else:
                counters["passes_incomplete"] += 1
        timeline.append({
            str(team_id): running[team_id].copy()
            for team_id in team_ids
        })

    return {
        "meta": {
            "frames": n,
            "fps": fps,
            "duration_s": round(n / fps, 2),
            "control_segments": len(segments),
            "skipped_transitions": dict(skipped),
            "config": asdict(cfg),
        },
        "teams": teams,
        "events": events,
        "timeline": timeline,
    }


def save_pass_stats(stats, path):
    import os

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Pass stats saved to {path}")


if __name__ == "__main__":
    # Offline use on a saved stub that already contains team + has_ball info:
    #   python pass_stats.py tracks.pkl out.json 25
    import pickle
    import sys

    stub, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
    with open(stub, "rb") as fh:
        tracks = pickle.load(fh)
    save_pass_stats(compute_pass_stats(tracks, fps=fps), out)