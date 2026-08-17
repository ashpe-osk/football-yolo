from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pickle
import os
import sys

sys.path.append("../")

from utils.bbox_utils import get_center_of_bbox, get_bbox_width


class Tracker:

    def __init__(self, model_path):

        # -------------------------------------------------
        # YOLOv26 - PLAYER DETECTOR + TRACKER
        # -------------------------------------------------

        self.model = YOLO(model_path)

        # -------------------------------------------------
        # YOLO11 - BALL + REFEREE DETECTOR
        # -------------------------------------------------

        self.ball_referee_model = YOLO(
            r"C:\football-yolo\models\ball_referee.pt"
        )

        # -------------------------------------------------
        # BoT-SORT + ReID configuration
        # -------------------------------------------------

        self.tracker_config = (
            r"C:\football-yolo\track_player\botsort_reid.yaml"
        )

        self.draw = sv.EllipseAnnotator()

        # -------------------------------------------------
        # GLOBAL ID STATE
        # -------------------------------------------------

        self.next_global_id = 1
        self.global_tracks = {}
        self.tracker_to_global = {}

        # -------------------------------------------------
        # MODIFIED: more tolerant thresholds
        # -------------------------------------------------
        self.global_max_age = 300
        self.max_position_distance = 0.5          # was 0.18
        self.max_appearance_distance = 0.6        # was 0.42
        self.max_combined_distance = 0.85         # was 0.62

    # =====================================================
    # DETECTION + TRACKING
    # =====================================================

    def detect_frames(self, frames):

        detections = []

        for frame_num, frame in enumerate(frames):

            player_results = self.model.track(
                source=frame,
                conf=0.1,
                tracker=self.tracker_config,
                persist=True,
                verbose=False
            )

            player_detection = player_results[0]

            ball_referee_results = self.ball_referee_model.predict(
                source=frame,
                conf=0.1,
                verbose=False
            )

            ball_referee_detection = ball_referee_results[0]

            detections.append({
                "players": player_detection,
                "ball_referee": ball_referee_detection
            })

            print(
                f"Frame {frame_num}: "
                f"YOLOv26 tracking complete"
            )

        return detections

    # =====================================================
    # GLOBAL ID ASSOCIATION HELPERS
    # =====================================================

    @staticmethod
    def _bbox_center(bbox):
        return np.array(
            [
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0
            ],
            dtype=np.float32
        )

    @staticmethod
    def _bbox_size(bbox):
        return (
            max(1.0, float(bbox[2] - bbox[0])),
            max(1.0, float(bbox[3] - bbox[1]))
        )

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def _appearance_feature(self, frame, bbox):
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(bbox[0])))
        y1 = max(0, min(h - 1, int(bbox[1])))
        x2 = max(0, min(w, int(bbox[2])))
        y2 = max(0, min(h, int(bbox[3])))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 6 or cw < 6:
            return None

        torso = crop[
            int(ch * 0.20):int(ch * 0.68),
            int(cw * 0.15):int(cw * 0.85)
        ]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    @staticmethod
    def _appearance_distance(a, b):
        if a is None or b is None:
            return 0.5
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

    def _position_distance(self, bbox, state, age):
        """
        Compute position distance with dynamic tolerance.
        The tolerance grows with age (uncertainty).
        """
        center = self._bbox_center(bbox)
        predicted = state["center"] + state.get("velocity", np.zeros(2, dtype=np.float32))

        # Scale the position tolerance linearly with age (up to a max factor)
        age_factor = min(1.0 + age * 0.02, 4.0)   # 2% per frame, cap at 4x
        dynamic_max_pos = self.max_position_distance * age_factor

        bw, bh = self._bbox_size(bbox)
        scale = max(30.0, float(np.sqrt(bw * bw + bh * bh)))
        raw_dist = float(np.linalg.norm(center - predicted) / scale)

        # Return the raw distance and the dynamic threshold for logging
        return raw_dist, dynamic_max_pos

    def _create_global_id(self, tracker_id, bbox, frame, frame_num, appearance=None):
        gid = self.next_global_id
        self.next_global_id += 1

        center = self._bbox_center(bbox)
        if appearance is None:
            appearance = self._appearance_feature(frame, bbox)

        self.global_tracks[gid] = {
            "center": center,
            "velocity": np.zeros(2, dtype=np.float32),
            "bbox": list(bbox),
            "appearance": appearance,
            "last_frame": frame_num,
            "last_tracker_id": tracker_id
        }
        self.tracker_to_global[tracker_id] = gid
        return gid

    # =====================================================
    # DUPLICATE GLOBAL ID CHECK
    # =====================================================

    def _find_duplicate_global(self, bbox, frame_num, used_global_ids):
        best_gid = None
        best_iou = 0.0

        for gid, state in self.global_tracks.items():
            if gid in used_global_ids:
                continue
            age = frame_num - state["last_frame"]
            if age < 0 or age > 20:   # extended from 5 to 20 frames
                continue

            iou = self._bbox_iou(bbox, state["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        if best_gid is not None and best_iou >= 0.40:
            return best_gid
        return None

    # =====================================================
    # GLOBAL ID ASSOCIATION 
    # =====================================================

    def _associate_global_ids(self, frames, tracks):

        self.next_global_id = 1
        self.global_tracks = {}
        self.tracker_to_global = {}

        for frame_num, frame_tracks in enumerate(tracks["players"]):

            if frame_num >= len(frames):
                break

            frame = frames[frame_num]
            detections = []

            # Prepare detections
            for tracker_id, player in frame_tracks.items():
                bbox = player["bbox"]
                detections.append({
                    "tracker_id": int(tracker_id),
                    "bbox": bbox,
                    "center": self._bbox_center(bbox),
                    "appearance": self._appearance_feature(frame, bbox)
                })

            assignments = {}
            used_global_ids = set()

            # 1) Preserve BoT-SORT continuity
            for det in detections:
                tid = det["tracker_id"]
                gid = self.tracker_to_global.get(tid)
                state = self.global_tracks.get(gid) if gid is not None else None
                if state is None:
                    continue
                age = frame_num - state["last_frame"]
                if age <= self.global_max_age and gid not in used_global_ids:
                    assignments[tid] = gid
                    used_global_ids.add(gid)

            # 2) Strong IoU duplicate check
            for det in detections:
                tid = det["tracker_id"]
                if tid in assignments:
                    continue
                duplicate_gid = self._find_duplicate_global(det["bbox"], frame_num, used_global_ids)
                if duplicate_gid is not None:
                    assignments[tid] = duplicate_gid
                    used_global_ids.add(duplicate_gid)
                    print(f"Frame {frame_num}: IoU duplicate: BoT {tid} -> Global {duplicate_gid}")

            # 3) Position + Appearance + Size matching (with dynamic thresholds)
            candidates = []
            for det in detections:
                tid = det["tracker_id"]
                if tid in assignments:
                    continue

                for gid, state in self.global_tracks.items():
                    if gid in used_global_ids:
                        continue

                    age = frame_num - state["last_frame"]
                    if age <= 0 or age > self.global_max_age:
                        continue

                    # Compute position distance with dynamic tolerance
                    pos, dynamic_pos_thresh = self._position_distance(det["bbox"], state, age)
                    # Normalize using the dynamic threshold (so we can compare)
                    pos_norm = min(pos / dynamic_pos_thresh, 2.0)

                    # Appearance distance (always with fixed threshold)
                    app = self._appearance_distance(det["appearance"], state["appearance"])
                    app_norm = min(app / self.max_appearance_distance, 2.0)

                    # Size similarity: ratio of areas or widths/heights
                    bw, bh = self._bbox_size(det["bbox"])
                    sw, sh = self._bbox_size(state["bbox"])
                    area_ratio = min(bw * bh / (sw * sh + 1e-6), (sw * sh) / (bw * bh + 1e-6))
                    size_sim = 1.0 - area_ratio   # 0 = identical

                    # Combine: position (0.5), appearance (0.35), size (0.15)
                    combined = 0.5 * pos_norm + 0.35 * app_norm + 0.15 * min(size_sim, 1.0)

                    # Use a combined threshold that is also dynamic (allow larger combined distance for older tracks)
                    dynamic_combined_thresh = self.max_combined_distance * (1.0 + min(age * 0.005, 0.5))

                    if combined <= dynamic_combined_thresh:
                        candidates.append((combined, tid, gid))

            # Sort by score (lowest = best)
            candidates.sort(key=lambda x: x[0])
            assigned_tids = set(assignments.keys())

            for score, tid, gid in candidates:
                if tid in assigned_tids:
                    continue
                if gid in used_global_ids:
                    continue
                assignments[tid] = gid
                assigned_tids.add(tid)
                used_global_ids.add(gid)
                # Log re-identification
                print(f"Frame {frame_num}: RE-ID: BoT {tid} -> Global {gid} (score={score:.3f})")

            # 4) Create new Global IDs for unmatched detections
            for det in detections:
                tid = det["tracker_id"]
                if tid in assignments:
                    continue
                gid = self._create_global_id(tid, det["bbox"], frame, frame_num, det["appearance"])
                assignments[tid] = gid
                used_global_ids.add(gid)
                print(f"Frame {frame_num}: NEW: BoT {tid} -> Global {gid}")

            # 5) Update global track states
            new_frame_tracks = {}
            for det in detections:
                tid = det["tracker_id"]
                gid = assignments[tid]
                state = self.global_tracks[gid]

                current_center = det["center"]
                velocity = current_center - state["center"]
                state["velocity"] = 0.7 * state.get("velocity", np.zeros(2, dtype=np.float32)) + 0.3 * velocity
                state["center"] = current_center
                state["bbox"] = list(det["bbox"])
                state["last_frame"] = frame_num
                state["last_tracker_id"] = tid

                if det["appearance"] is not None:
                    if state.get("appearance") is None:
                        state["appearance"] = det["appearance"]
                    else:
                        state["appearance"] = (0.8 * state["appearance"] + 0.2 * det["appearance"]).astype(np.float32)

                self.tracker_to_global[tid] = gid
                new_frame_tracks[gid] = {
                    "bbox": det["bbox"],
                    "tracker_id": tid,
                    "global_id": gid
                }

            tracks["players"][frame_num] = new_frame_tracks

            # Log summary for this frame
            print(f"Frame {frame_num}: active Global IDs = {list(new_frame_tracks.keys())}")

        return tracks

    # =====================================================
    # GET TRACKS
    # =====================================================

    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)

            # Check if pickle already has Global IDs
            needs_global_ids = False
            for frame_tracks in tracks.get("players", []):
                if frame_tracks:
                    first_player = next(iter(frame_tracks.values()))
                    needs_global_ids = "global_id" not in first_player
                    break

            if needs_global_ids:
                print("Existing stub has no Global IDs. Associating Global IDs...")
                tracks = self._associate_global_ids(frames, tracks)
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)
                print(f"Updated stub with Global IDs: {stub_path}")
            return tracks

        # Detect + track
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):

            # Players
            player_detection = detection["players"]
            tracks["players"].append({})
            if player_detection.boxes is not None and player_detection.boxes.id is not None:
                boxes = player_detection.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().tolist()
                    track_id = int(boxes.id[i].cpu().item())
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # Ball + Referee
            ball_referee_detection = detection["ball_referee"]
            ball_referee_supervision = sv.Detections.from_ultralytics(ball_referee_detection)
            referee_indices = []
            ball_indices = []

            for i, class_id in enumerate(ball_referee_supervision.class_id):
                class_name = ball_referee_detection.names[class_id]
                if class_name == "Referee":
                    referee_indices.append(i)
                elif class_name == "Ball":
                    ball_indices.append(i)

            # Referees
            tracks["referees"].append({})
            if len(referee_indices) > 0:
                for i in referee_indices:
                    bbox = ball_referee_supervision.xyxy[i].tolist()
                    referee_id = i + 1
                    tracks["referees"][frame_num][referee_id] = {"bbox": bbox}

            # Ball
            tracks["ball"].append({})
            if len(ball_indices) > 0:
                best_ball_index = max(ball_indices, key=lambda i: ball_referee_supervision.confidence[i])
                bbox = ball_referee_supervision.xyxy[best_ball_index].tolist()
                tracks["ball"][frame_num][1] = {"bbox": bbox}

            print(
                f"Frame {frame_num}: "
                f"{len(tracks['players'][frame_num])} players, "
                f"{len(tracks['referees'][frame_num])} referees, "
                f"{len(tracks['ball'][frame_num])} ball"
            )

        # Convert BoT-SORT IDs -> Global IDs
        tracks = self._associate_global_ids(frames, tracks)

        # Save tracks
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"Tracks saved to {stub_path}")

        return tracks

    # =====================================================
    # DRAWING 
    # =====================================================

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        ellipse_x_radius = int(width)
        ellipse_y_radius = int(0.35 * width)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(ellipse_x_radius, ellipse_y_radius),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rect_width = 40
            rect_height = 20
            ellipse_bottom = y2 + ellipse_y_radius
            y1_rect = ellipse_bottom - rect_height // 2
            y2_rect = y1_rect + rect_height
            x1_rect = x_center - rect_width // 2
            x2_rect = x_center + rect_width // 2

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            text = str(track_id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x_center - text_size[0] // 2
            text_y = y1_rect + (rect_height + text_size[1]) // 2
            cv2.putText(frame, text, (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames