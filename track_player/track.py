from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.append("../")

from utils.bbox_utils import get_center_of_bbox, get_bbox_width


class Tracker:

    def __init__(self, model_path):

        # =================================================
        # YOLOv26 - PLAYER DETECTOR + TRACKER
        # =================================================

        self.model = YOLO(model_path)

        # =================================================
        # YOLO11 - BALL + REFEREE DETECTOR
        # =================================================

        self.ball_referee_model = YOLO(
            r"C:\football-yolo\models\ball_referee.pt"
        )

        # =================================================
        # BoT-SORT + ReID
        # =================================================

        self.tracker_config = (
            r"C:\football-yolo\track_player\botsort_reid.yaml"
        )

        self.draw = sv.EllipseAnnotator()

        # =================================================
        # GLOBAL ID STATE
        # =================================================

        self.next_global_id = 1
        self.global_tracks = {}
        self.tracker_to_global = {}

        # =================================================
        # GLOBAL TRACK SETTINGS
        # =================================================

        self.global_max_age = 300
        self.max_position_distance = 0.35
        self.max_appearance_distance = 0.65
        self.max_combined_distance = 1.0

        # =================================================
        # CAMERA MOTION / RECOVERY SETTINGS
        # =================================================

        self.recovery_duration = 10
        self.recovery_until = -1
        self.previous_detection_count = 0
        self.camera_transforms = {}
        self.gmc_max_corners = 300
        self.gmc_min_points = 12
        self.debug_camera = True

        # =================================================
        # CAMERA MOVEMENT DATA
        # =================================================
        self.camera_movement_per_frame = []   # will store (tx, ty) per frame

    # =====================================================
    # BALL INTERPOLATION
    # =====================================================

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

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
            print(f"Frame {frame_num}: YOLOv26 tracking complete")
        return detections

    # =====================================================
    # CAMERA MOVEMENT COMPUTATION (standalone)
    # =====================================================

    def compute_camera_movement(self, frames):
        """
        Compute per‑frame camera translation (tx, ty) using optical flow.
        Returns list of (tx, ty) for each frame (first frame is (0,0)).
        """
        if not frames:
            return []

        movement = [(0.0, 0.0)]  # first frame has no movement

        previous_frame = frames[0]
        for i in range(1, len(frames)):
            matrix, _, _ = self._estimate_camera_motion(previous_frame, frames[i])
            tx = float(matrix[0, 2])
            ty = float(matrix[1, 2])
            movement.append((tx, ty))
            previous_frame = frames[i]

        self.camera_movement_per_frame = movement
        return movement

    # =====================================================
    # BASIC GEOMETRY HELPERS
    # =====================================================

    @staticmethod
    def _bbox_center(bbox):
        return np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float32
        )

    @staticmethod
    def _get_foot_position(bbox):
        """
        Compute the foot position (bottom‑center) of a bounding box.
        Returns (cx, y_bottom) as integers.
        """
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    @staticmethod
    def _bbox_size(bbox):
        return (
            max(1.0, float(bbox[2] - bbox[0])),
            max(1.0, float(bbox[3] - bbox[1]))
        )

    @staticmethod
    def _bbox_area(bbox):
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        return width * height

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

    # =====================================================
    # CAMERA MOTION ESTIMATION
    # =====================================================

    def _estimate_camera_motion(self, previous_frame, current_frame):
        height, width = previous_frame.shape[:2]
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.gmc_max_corners,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7
        )
        if previous_points is None:
            return np.eye(2, 3, dtype=np.float32), 0.0, 0
        current_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if current_points is None:
            return np.eye(2, 3, dtype=np.float32), 0.0, 0
        status = status.reshape(-1)
        good_previous = previous_points.reshape(-1, 2)[status == 1]
        good_current = current_points.reshape(-1, 2)[status == 1]
        if len(good_previous) < self.gmc_min_points:
            return np.eye(2, 3, dtype=np.float32), 0.0, len(good_previous)
        matrix, inliers = cv2.estimateAffinePartial2D(
            good_previous,
            good_current,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=2000,
            confidence=0.99
        )
        if matrix is None:
            return np.eye(2, 3, dtype=np.float32), 0.0, len(good_previous)
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        translation = np.sqrt(tx * tx + ty * ty)
        diagonal = np.sqrt(float(width * width) + float(height * height))
        normalized_motion = translation / max(diagonal, 1.0)
        return (matrix.astype(np.float32), normalized_motion, len(good_previous))

    @staticmethod
    def _transform_point(point, matrix):
        x = float(point[0])
        y = float(point[1])
        transformed_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        transformed_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return np.array([transformed_x, transformed_y], dtype=np.float32)

    @staticmethod
    def _transform_vector(vector, matrix):
        x = float(vector[0])
        y = float(vector[1])
        transformed_x = matrix[0, 0] * x + matrix[0, 1] * y
        transformed_y = matrix[1, 0] * x + matrix[1, 1] * y
        return np.array([transformed_x, transformed_y], dtype=np.float32)

    # =====================================================
    # APPEARANCE FEATURE
    # =====================================================

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
        if ch < 8 or cw < 8:
            return None
        torso = crop[
            int(ch * 0.18):int(ch * 0.72),
            int(cw * 0.12):int(cw * 0.88)
        ]
        if torso.size == 0:
            return None
        torso = cv2.resize(torso, (64, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        gray_mean = np.mean(gray)
        gray_std = np.std(gray) + 1e-6
        gray = (gray - gray_mean) / gray_std
        gray = np.clip(gray, -3.0, 3.0).flatten()
        feature = np.concatenate([hist.astype(np.float32), gray.astype(np.float32)])
        norm = np.linalg.norm(feature)
        if norm <= 1e-6:
            return None
        feature = feature / norm
        return feature.astype(np.float32)

    @staticmethod
    def _appearance_distance(a, b):
        if a is None or b is None:
            return 0.5
        distance = np.linalg.norm(a - b)
        return float(min(distance / 2.0, 1.0))

    def _size_distance(self, bbox_a, bbox_b):
        area_a = self._bbox_area(bbox_a)
        area_b = self._bbox_area(bbox_b)
        ratio = min(area_a / (area_b + 1e-6), area_b / (area_a + 1e-6))
        return float(1.0 - ratio)

    # =====================================================
    # PREDICT GLOBAL TRACK POSITION
    # =====================================================

    def _predict_global_position(self, state, current_frame_num, current_camera_matrix):
        center = state["predicted_center"].copy()
        velocity = state.get("velocity", np.zeros(2, dtype=np.float32))
        camera_center = self._transform_point(center, current_camera_matrix)
        transformed_velocity = self._transform_vector(velocity, current_camera_matrix)
        velocity_norm = np.linalg.norm(transformed_velocity)
        if velocity_norm > 120:
            transformed_velocity = (transformed_velocity / velocity_norm) * 120.0
        predicted = camera_center + transformed_velocity
        return predicted

    # =====================================================
    # CREATE GLOBAL ID
    # =====================================================

    def _create_global_id(self, tracker_id, bbox, frame, frame_num, appearance=None):
        gid = self.next_global_id
        self.next_global_id += 1
        center = self._bbox_center(bbox)
        if appearance is None:
            appearance = self._appearance_feature(frame, bbox)
        self.global_tracks[gid] = {
            "center": center.copy(),
            "predicted_center": center.copy(),
            "velocity": np.zeros(2, dtype=np.float32),
            "bbox": list(bbox),
            "appearance": appearance,
            "last_frame": frame_num,
            "last_tracker_id": tracker_id
        }
        if tracker_id not in self.tracker_to_global:
            self.tracker_to_global[tracker_id] = gid
        return gid

    # =====================================================
    # GLOBAL ID ASSOCIATION
    # =====================================================

    def _associate_global_ids(self, frames, tracks):
        self.next_global_id = 1
        self.global_tracks = {}
        self.tracker_to_global = {}
        self.camera_transforms = {}
        self.recovery_until = -1
        self.previous_detection_count = 0
        previous_frame = None

        # Reset camera movement storage for this run
        self.camera_movement_per_frame = []

        for frame_num, frame_tracks in enumerate(tracks["players"]):
            if frame_num >= len(frames):
                break
            frame = frames[frame_num]

            if previous_frame is None:
                camera_matrix = np.eye(2, 3, dtype=np.float32)
                camera_motion = 0.0
                flow_points = 0
                self.camera_movement_per_frame.append((0.0, 0.0))  # first frame
            else:
                camera_matrix, camera_motion, flow_points = self._estimate_camera_motion(previous_frame, frame)
                tx = float(camera_matrix[0, 2])
                ty = float(camera_matrix[1, 2])
                self.camera_movement_per_frame.append((tx, ty))

            self.camera_transforms[frame_num] = camera_matrix
            current_detection_count = len(frame_tracks)

            detection_drop = False
            if self.previous_detection_count >= 6:
                if current_detection_count < (self.previous_detection_count * 0.65):
                    detection_drop = True

            major_camera_motion = (camera_motion >= 0.018)
            severe_camera_motion = (camera_motion >= 0.035)

            if major_camera_motion or detection_drop:
                self.recovery_until = max(self.recovery_until, frame_num + self.recovery_duration)
                if self.debug_camera:
                    print(f"Frame {frame_num}: CAMERA TRANSITION (motion={camera_motion:.3f}, detections={current_detection_count}, previous={self.previous_detection_count})")

            recovery_mode = (frame_num <= self.recovery_until)

            detections = []
            for tracker_id, player in frame_tracks.items():
                bbox = player["bbox"]
                detections.append({
                    "tracker_id": int(tracker_id),
                    "bbox": bbox,
                    "center": self._bbox_center(bbox),
                    "appearance": self._appearance_feature(frame, bbox)
                })

            predicted_positions = {}
            for gid, state in self.global_tracks.items():
                age = frame_num - state["last_frame"]
                if age < 0 or age > self.global_max_age:
                    continue
                predicted = self._predict_global_position(state, frame_num, camera_matrix)
                predicted_positions[gid] = predicted
                state["predicted_center"] = predicted

            assignments = {}
            used_global_ids = set()

            # 6. PRESERVE DIRECT BoT-SORT CONTINUITY (FORCED)
            for det in detections:
                tid = det["tracker_id"]
                gid = self.tracker_to_global.get(tid)
                if gid is None:
                    continue
                state = self.global_tracks.get(gid)
                if state is None:
                    continue
                age = frame_num - state["last_frame"]
                if age <= self.global_max_age and gid not in used_global_ids:
                    assignments[tid] = gid
                    used_global_ids.add(gid)

            # 7. BUILD MATCHING CANDIDATES
            candidates = []
            for det in detections:
                tid = det["tracker_id"]
                if tid in assignments:
                    continue
                bbox = det["bbox"]
                center = det["center"]
                appearance = det["appearance"]
                for gid, state in self.global_tracks.items():
                    if gid in used_global_ids:
                        continue
                    age = frame_num - state["last_frame"]
                    if age <= 0 or age > self.global_max_age:
                        continue
                    predicted = predicted_positions.get(gid, state["center"])
                    distance = np.linalg.norm(center - predicted)
                    bw, bh = self._bbox_size(bbox)
                    player_scale = max(35.0, float(np.sqrt(bw * bw + bh * bh)))
                    if recovery_mode:
                        position_limit = player_scale * (3.0 + min(age * 0.20, 6.0))
                    else:
                        position_limit = player_scale * (2.0 + min(age * 0.12, 4.0))
                    position_score = min(distance / max(position_limit, 1.0), 2.0)
                    appearance_score = self._appearance_distance(appearance, state.get("appearance"))
                    size_score = self._size_distance(bbox, state["bbox"])
                    predicted_bbox = [
                        predicted[0] - (state["bbox"][2] - state["bbox"][0]) / 2.0,
                        predicted[1] - (state["bbox"][3] - state["bbox"][1]) / 2.0,
                        predicted[0] + (state["bbox"][2] - state["bbox"][0]) / 2.0,
                        predicted[1] + (state["bbox"][3] - state["bbox"][1]) / 2.0
                    ]
                    iou = self._bbox_iou(bbox, predicted_bbox)
                    iou_score = 1.0 - iou

                    if not recovery_mode:
                        combined = 0.50 * position_score + 0.30 * appearance_score + 0.10 * size_score + 0.10 * iou_score
                        threshold = self.max_combined_distance * (1.0 + min(age * 0.015, 0.80))
                    else:
                        combined = 0.35 * position_score + 0.45 * appearance_score + 0.10 * size_score + 0.10 * iou_score
                        threshold = 1.35 if not severe_camera_motion else 1.50

                    appearance_allowed = (appearance_score <= 1.0)
                    if combined <= threshold and appearance_allowed:
                        candidates.append((combined, tid, gid, position_score, appearance_score, size_score, iou_score))

            candidates.sort(key=lambda x: x[0])
            assigned_tids = set(assignments.keys())

            # 9. ONE-TO-ONE ASSIGNMENT
            for candidate in candidates:
                score, tid, gid, position_score, appearance_score, size_score, iou_score = candidate
                if tid in assigned_tids or gid in used_global_ids:
                    continue
                existing_gid = self.tracker_to_global.get(tid)
                if existing_gid is not None and existing_gid != gid:
                    continue
                assignments[tid] = gid
                assigned_tids.add(tid)
                used_global_ids.add(gid)
                if recovery_mode:
                    print(f"Frame {frame_num}: CAMERA-RECOVERY: BoT {tid} -> Global {gid} (score={score:.3f}, pos={position_score:.3f}, app={appearance_score:.3f})")
                else:
                    print(f"Frame {frame_num}: RE-ID: BoT {tid} -> Global {gid} (score={score:.3f})")

            # 9.5 FALLBACK ASSIGNMENT
            for det in detections:
                tid = det["tracker_id"]
                if tid in assigned_tids:
                    continue

                existing_gid = self.tracker_to_global.get(tid)
                if existing_gid is not None:
                    state = self.global_tracks.get(existing_gid)
                    if state is not None:
                        age = frame_num - state["last_frame"]
                        if age <= self.global_max_age and existing_gid not in used_global_ids:
                            assignments[tid] = existing_gid
                            assigned_tids.add(tid)
                            used_global_ids.add(existing_gid)
                            if recovery_mode:
                                print(f"Frame {frame_num}: FORCED (existing mapping): BoT {tid} -> Global {existing_gid} (recovery)")
                            else:
                                print(f"Frame {frame_num}: FORCED (existing mapping): BoT {tid} -> Global {existing_gid}")
                            continue

                best_score = float('inf')
                best_gid = None
                best_appearance = 1.0
                best_position = 2.0

                bbox = det["bbox"]
                center = det["center"]
                appearance = det["appearance"]

                for gid, state in self.global_tracks.items():
                    if gid in used_global_ids:
                        continue
                    age = frame_num - state["last_frame"]
                    if age <= 0 or age > self.global_max_age:
                        continue

                    predicted = predicted_positions.get(gid, state["center"])
                    abs_distance = np.linalg.norm(center - predicted)

                    bw, bh = self._bbox_size(bbox)
                    player_scale = max(35.0, float(np.sqrt(bw * bw + bh * bh)))
                    if recovery_mode:
                        position_limit = player_scale * (3.0 + min(age * 0.20, 6.0))
                    else:
                        position_limit = player_scale * (2.0 + min(age * 0.12, 4.0))
                    position_score = min(abs_distance / max(position_limit, 1.0), 2.0)

                    appearance_score = self._appearance_distance(appearance, state.get("appearance"))
                    size_score = self._size_distance(bbox, state["bbox"])

                    predicted_bbox = [
                        predicted[0] - (state["bbox"][2] - state["bbox"][0]) / 2.0,
                        predicted[1] - (state["bbox"][3] - state["bbox"][1]) / 2.0,
                        predicted[0] + (state["bbox"][2] - state["bbox"][0]) / 2.0,
                        predicted[1] + (state["bbox"][3] - state["bbox"][1]) / 2.0
                    ]
                    iou = self._bbox_iou(bbox, predicted_bbox)
                    iou_score = 1.0 - iou

                    combined = 0.50 * position_score + 0.30 * appearance_score + 0.10 * size_score + 0.10 * iou_score

                    if combined < best_score:
                        best_score = combined
                        best_gid = gid
                        best_appearance = appearance_score
                        best_position = position_score

                max_score = 2.0 if recovery_mode else 1.5
                max_position = 2.0 if recovery_mode else 1.5
                max_appearance = 0.60

                if (
                    best_gid is not None
                    and best_score < max_score
                    and best_appearance < max_appearance
                    and best_position < max_position
                ):
                    assignments[tid] = best_gid
                    assigned_tids.add(tid)
                    used_global_ids.add(best_gid)
                    if tid not in self.tracker_to_global:
                        self.tracker_to_global[tid] = best_gid
                    if recovery_mode:
                        print(f"Frame {frame_num}: CAMERA-RECOVERY FALLBACK: BoT {tid} -> Global {best_gid} (score={best_score:.3f}, app={best_appearance:.3f})")
                    else:
                        print(f"Frame {frame_num}: FALLBACK: BoT {tid} -> Global {best_gid} (score={best_score:.3f})")

            # 10. CREATE NEW GLOBAL IDS
            for det in detections:
                tid = det["tracker_id"]
                if tid in assignments:
                    continue
                gid = self._create_global_id(tid, det["bbox"], frame, frame_num, det["appearance"])
                assignments[tid] = gid
                used_global_ids.add(gid)
                print(f"Frame {frame_num}: NEW: BoT {tid} -> Global {gid}")

            # 11. UPDATE GLOBAL TRACK STATES
            new_frame_tracks = {}
            for det in detections:
                tid = det["tracker_id"]
                gid = assignments[tid]
                state = self.global_tracks[gid]
                current_center = det["center"]
                old_predicted = state.get("predicted_center", state["center"])
                velocity = current_center - old_predicted
                velocity_norm = np.linalg.norm(velocity)
                if velocity_norm > 100:
                    velocity = (velocity / velocity_norm) * 100.0
                old_velocity = state.get("velocity", np.zeros(2, dtype=np.float32))
                state["velocity"] = 0.65 * old_velocity + 0.35 * velocity
                state["center"] = current_center.copy()
                state["predicted_center"] = current_center.copy()
                state["bbox"] = list(det["bbox"])
                state["last_frame"] = frame_num
                state["last_tracker_id"] = tid

                if det["appearance"] is not None:
                    if state.get("appearance") is None:
                        state["appearance"] = det["appearance"]
                    else:
                        updated = 0.90 * state["appearance"] + 0.10 * det["appearance"]
                        norm = np.linalg.norm(updated)
                        if norm > 1e-6:
                            updated = updated / norm
                        state["appearance"] = updated.astype(np.float32)

                if tid not in self.tracker_to_global:
                    self.tracker_to_global[tid] = gid
                else:
                    if self.tracker_to_global[tid] != gid:
                        print(f"WARNING: BoT ID {tid} was previously mapped to Global {self.tracker_to_global[tid]}, but now assigned to {gid}. Keeping old mapping.")
                        gid = self.tracker_to_global[tid]

                new_frame_tracks[gid] = {
                    "bbox": det["bbox"],
                    "tracker_id": tid,
                    "global_id": gid
                }

            tracks["players"][frame_num] = new_frame_tracks

            mode = "CAMERA-RECOVERY" if recovery_mode else "NORMAL"
            print(f"Frame {frame_num}: {mode} | Global IDs = {list(new_frame_tracks.keys())}")

            self.previous_detection_count = current_detection_count
            previous_frame = frame

        return tracks

    # =====================================================
    # ADD POSITIONS TO TRACKS
    # =====================================================

    def add_positions_to_tracks(self, tracks):
        """
        Adds a 'position' field to every tracked object in all frames.
        - For the ball: center of the bounding box.
        - For players and referees: foot position (bottom‑center of the bbox).
        """
        for object_name, object_tracks in tracks.items():
            if object_name not in ["players", "referees", "ball"]:
                continue
            for frame_num, track_dict in enumerate(object_tracks):
                for track_id, track_info in track_dict.items():
                    bbox = track_info['bbox']
                    if object_name == 'ball':
                        # Use center for ball
                        center = self._bbox_center(bbox)
                        position = (center[0], center[1])  # as tuple/list
                    else:
                        # Use foot position for players and referees
                        position = self._get_foot_position(bbox)
                    track_info['position'] = position
        return tracks

    # =====================================================
    # DRAW CAMERA MOVEMENT
    # =====================================================

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)
        return output_frames

    # =====================================================
    # GET TRACKS
    # =====================================================

    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
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

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            player_detection = detection["players"]
            tracks["players"].append({})
            if player_detection.boxes is not None and player_detection.boxes.id is not None:
                boxes = player_detection.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().tolist()
                    track_id = int(boxes.id[i].cpu().item())
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

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

            tracks["referees"].append({})
            if len(referee_indices) > 0:
                for i in referee_indices:
                    bbox = ball_referee_supervision.xyxy[i].tolist()
                    referee_id = i + 1
                    tracks["referees"][frame_num][referee_id] = {"bbox": bbox}

            tracks["ball"].append({})
            if len(ball_indices) > 0:
                best_ball_index = max(ball_indices, key=lambda i: ball_referee_supervision.confidence[i])
                bbox = ball_referee_supervision.xyxy[best_ball_index].tolist()
                tracks["ball"][frame_num][1] = {"bbox": bbox}

            print(f"Frame {frame_num}: {len(tracks['players'][frame_num])} players, {len(tracks['referees'][frame_num])} referees, {len(tracks['ball'][frame_num])} ball")

        tracks = self._associate_global_ids(frames, tracks)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"Tracks saved to {stub_path}")

        return tracks

    # =====================================================
    # DRAW ELLIPSE
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
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total_frames = team1_num_frames + team2_num_frames
        if total_frames == 0:
            team1_percent = 0.0
            team2_percent = 0.0
        else:
            team1_percent = team1_num_frames / total_frames * 100
            team2_percent = team2_num_frames / total_frames * 100

        cv2.putText(frame, "POSSESSION", (1400, 885), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, f"Team 1 Possession: {team1_percent:.2f}%", (1400, 925), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Possession: {team2_percent:.2f}%", (1400, 965), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames