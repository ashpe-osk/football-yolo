from dataclasses import dataclass, fields
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

from utils.bbox_utils import get_center_of_bbox, get_bbox_width


@dataclass(frozen=True)
class TrackerConfig:
    max_global_ids: int = 22
    global_max_age: int = 300
    tid_mapping_max_age: int = 30
    min_confirm_frames: int = 3
    tentative_max_gap: int = 2
    match_acceptance_cost: float = 0.72
    match_ambiguity_margin: float = 0.07
    forced_position_base_px: float = 45.0
    forced_max_speed_px_per_frame: float = 22.0
    forced_appearance_limit: float = 0.68
    position_gate_base_px: float = 55.0
    position_gate_age_speed_px_per_frame: float = 10.0
    position_gate_max_age_expansion_px: float = 300.0
    appearance_reject_limit: float = 0.68
    appearance_update_max_distance: float = 0.55
    position_score_weight: float = 0.55
    appearance_score_weight: float = 0.30
    size_score_weight: float = 0.05
    iou_score_weight: float = 0.10
    appearance_weight_min: float = 0.12
    appearance_weight_decay_per_frame: float = 0.001
    recovery_position_gate_bonus_px: float = 80.0
    velocity_ema_alpha: float = 0.35
    velocity_decay_per_frame: float = 0.92
    max_velocity_px_per_frame: float = 35.0
    max_extrapolation_px: float = 220.0
    max_new_global_ids_per_frame: int = 3
    camera_mask_padding_px: int = 12
    camera_motion_recovery_threshold: float = 0.018
    camera_scene_change_inlier_ratio: float = 0.20
    camera_minimum_inlier_ratio: float = 0.35
    camera_minimum_feature_points: int = 12
    camera_maximum_feature_points: int = 300
    recovery_enter_frames: int = 2
    recovery_exit_frames: int = 5
    pitch_min_inliers: int = 6
    pitch_min_inlier_ratio: float = 0.35
    pitch_min_area_fraction: float = 0.01
    pitch_max_area_fraction: float = 0.95
    pitch_min_aspect_ratio: float = 0.45
    pitch_max_aspect_ratio: float = 6.0
    pitch_keypoint_confidence_threshold: float = 0.5
    pitch_touchline_tolerance_px: float = 18.0
    pitch_min_keypoints: int = 4
    pitch_max_unexplained_shift_px: float = 100.0
    pitch_smoothing_alpha: float = 0.65
    pitch_max_reuse_frames: int = 30
    pitch_ransac_reprojection_px: float = 5.0
    pitch_ransac_reprojection_cm: float = 500.0
    field_inference_interval_frames: int = 10
    field_inference_confidence: float = 0.30
    player_detection_confidence: float = 0.10
    referee_ball_detection_confidence: float = 0.10
    camera_feature_quality_level: float = 0.01
    camera_feature_min_distance_px: float = 15.0
    camera_feature_block_size: int = 7
    camera_flow_window_px: int = 21
    camera_flow_max_level: int = 3
    camera_flow_max_iterations: int = 30
    camera_flow_epsilon: float = 0.01
    camera_affine_ransac_reprojection_px: float = 4.0
    camera_affine_ransac_max_iterations: int = 2000
    camera_affine_ransac_confidence: float = 0.99
    ball_max_interpolation_gap_frames: int = 12
    ball_box_max_size_change_ratio: float = 2.5
    debug_invariants: bool = True

    @classmethod
    def from_environment(cls):
        defaults = cls()
        values = {}
        for field in fields(cls):
            environment_name = f"TRACKER_{field.name.upper()}"
            raw_value = os.getenv(environment_name)
            if raw_value is None:
                continue

            default_value = getattr(defaults, field.name)
            try:
                if isinstance(default_value, bool):
                    values[field.name] = raw_value.strip().lower() in {
                        "1", "true", "yes", "on"
                    }
                elif isinstance(default_value, int):
                    values[field.name] = int(raw_value)
                elif isinstance(default_value, float):
                    values[field.name] = float(raw_value)
                else:
                    values[field.name] = raw_value
            except ValueError as exc:
                raise ValueError(
                    f"{environment_name} must be a valid "
                    f"{type(default_value).__name__}."
                ) from exc

        legacy_environment_names = {
            "max_global_ids": "MAX_GLOBAL_IDS",
            "global_max_age": "GLOBAL_MAX_AGE",
            "tid_mapping_max_age": "TID_MAPPING_MAX_AGE",
            "min_confirm_frames": "MIN_CONFIRM_FRAMES",
            "pitch_keypoint_confidence_threshold": (
                "PITCH_KEYPOINT_CONFIDENCE_THRESHOLD"
            ),
            "pitch_touchline_tolerance_px": "TOUCHLINE_TOLERANCE",
            "debug_invariants": "TRACKER_DEBUG_INVARIANTS"
        }
        for field_name, environment_name in legacy_environment_names.items():
            raw_value = os.getenv(environment_name)
            if raw_value is None:
                continue
            default_value = getattr(defaults, field_name)
            try:
                if isinstance(default_value, bool):
                    values[field_name] = raw_value.strip().lower() in {
                        "1", "true", "yes", "on"
                    }
                elif isinstance(default_value, int):
                    values[field_name] = int(raw_value)
                else:
                    values[field_name] = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{environment_name} must be a valid "
                    f"{type(default_value).__name__}."
                ) from exc

        for field_name in (
            "max_global_ids",
            "global_max_age",
            "tid_mapping_max_age",
            "min_confirm_frames",
            "max_new_global_ids_per_frame",
            "pitch_min_keypoints",
            "camera_minimum_feature_points",
            "camera_maximum_feature_points",
        ):
            value = values.get(field_name, getattr(defaults, field_name))
            if value < 1:
                raise ValueError(f"{field_name} must be at least 1.")

        return cls(**values)


class Tracker:

    PITCH_LENGTH_CM = 12000
    PITCH_WIDTH_CM = 7000
    FIELD_KEYPOINTS = np.array([
        (0, 0), (0, 1450), (0, 2584), (0, 4416), (0, 5550), (0, 7000),
        (550, 2584), (550, 4416), (1100, 3500),
        (2015, 1450), (2015, 2584), (2015, 4416), (2015, 5550),
        (6000, 0), (6000, 2585), (6000, 4415), (6000, 7000),
        (9985, 1450), (9985, 2584), (9985, 4416), (9985, 5550),
        (10900, 3500), (11450, 2584), (11450, 4416),
        (12000, 0), (12000, 1450), (12000, 2584), (12000, 4416),
        (12000, 5550), (12000, 7000), (5085, 3500), (6915, 3500),
    ], dtype=np.float32)
    PITCH_BOUNDARY = np.array([
        (0, 0),
        (PITCH_LENGTH_CM, 0),
        (PITCH_LENGTH_CM, PITCH_WIDTH_CM),
        (0, PITCH_WIDTH_CM),
    ], dtype=np.float32)
    REFEREE_COLOR = (255, 255, 0)
    UNASSIGNED_PLAYER_COLOR = (180, 180, 180)
    PITCH_FILTER_VERSION = 2

    def __init__(self, model_path):

        load_dotenv()

        # YOLOv26 - PLAYER DETECTOR + TRACKER
        

        self.model = YOLO(model_path)

        
        # YOLO11 - BALL + REFEREE DETECTOR
        

        self.ball_referee_model = YOLO(
            r"C:\football-yolo\models\ball_referee.pt"
        )

        
        # BoT-SORT + ReID
        

        self.tracker_config = (
            r"C:\football-yolo\track_player\botsort_reid.yaml"
        )

        self.draw = sv.EllipseAnnotator()

        
        # GLOBAL ID STATE
        

        self.config = TrackerConfig.from_environment()
        self.next_global_id = 1
        self.global_tracks = {}
        self.tid_to_gid = {}
        self.gid_to_tid = {}
        self.tid_last_seen = {}
        self.tentative_tracks = {}

        
        # GLOBAL TRACK SETTINGS
        

        self.global_max_age = self.config.global_max_age
        self.max_position_distance = 0.35
        self.max_appearance_distance = 0.65
        self.max_combined_distance = 1.0

        
        # CAMERA MOTION / RECOVERY SETTINGS
        

        self.recovery_duration = 10
        self.recovery_active = False
        self.recovery_trigger_count = 0
        self.recovery_quiet_count = 0
        self.camera_transforms = {}
        self.camera_step_transforms = {}
        self.gmc_max_corners = self.config.camera_maximum_feature_points
        self.gmc_min_points = self.config.camera_minimum_feature_points
        self.debug_camera = self.config.debug_invariants

        # PITCH FILTERING / PLAYER VALIDATION SETTINGS
        self.field_detection_model = None
        self.pitch_keypoint_confidence_threshold = (
            self.config.pitch_keypoint_confidence_threshold
        )
        self.touchline_tolerance = self.config.pitch_touchline_tolerance_px
        self.min_pitch_keypoints = self.config.pitch_min_keypoints

        
        # CAMERA MOVEMENT DATA
        
        self.camera_movement_per_frame = []   # will store (tx, ty) per frame
        self.camera_transform_per_frame = []
        self.camera_inlier_ratio_per_frame = []
        self.last_good_pitch_polygon = None
        self.last_good_pitch_frame = -1
        self.pitch_status_per_frame = []
        self.last_pitch_status = "missing"

   
    # BALL INTERPOLATION
   

    def interpolate_ball_positions(
        self,
        ball_positions,
        max_gap=12
    ):
        frame_count = len(ball_positions)
        if frame_count == 0:
            return []
        if len(self.camera_transform_per_frame) != frame_count:
            raise RuntimeError(
                "Camera transforms must be computed before ball interpolation."
            )

        observed = {}
        for frame_num, frame_tracks in enumerate(ball_positions):
            bbox = frame_tracks.get(1, {}).get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            observed[frame_num] = bbox

        if not observed:
            return [{} for _ in range(frame_count)]

        observed_frames = sorted(observed)
        stabilized_centers = {}
        for frame_num, bbox in observed.items():
            center = np.array(
                [[[(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]]],
                dtype=np.float32
            )
            image_to_reference = np.linalg.inv(
                self.camera_transform_per_frame[frame_num]
            )
            stabilized_centers[frame_num] = cv2.perspectiveTransform(
                center, image_to_reference
            ).reshape(2)

        output = [{} for _ in range(frame_count)]
        for frame_num, bbox in observed.items():
            output[frame_num] = {1: {"bbox": bbox.tolist()}}

        for left_frame, right_frame in zip(observed_frames, observed_frames[1:]):
            missing_count = right_frame - left_frame - 1
            if missing_count <= 0 or missing_count > max_gap:
                continue

            left_bbox = observed[left_frame]
            right_bbox = observed[right_frame]
            left_center = stabilized_centers[left_frame]
            right_center = stabilized_centers[right_frame]
            left_size = np.array(
                [left_bbox[2] - left_bbox[0], left_bbox[3] - left_bbox[1]],
                dtype=np.float32
            )
            right_size = np.array(
                [right_bbox[2] - right_bbox[0], right_bbox[3] - right_bbox[1]],
                dtype=np.float32
            )

            for frame_num in range(left_frame + 1, right_frame):
                fraction = (frame_num - left_frame) / (right_frame - left_frame)
                stabilized_center = (
                    left_center * (1.0 - fraction) + right_center * fraction
                )
                reference_to_image = self.camera_transform_per_frame[frame_num]
                image_center = cv2.perspectiveTransform(
                    stabilized_center.reshape(1, 1, 2),
                    reference_to_image
                ).reshape(2)
                size = left_size * (1.0 - fraction) + right_size * fraction
                interpolated_bbox = [
                    float(image_center[0] - size[0] / 2.0),
                    float(image_center[1] - size[1] / 2.0),
                    float(image_center[0] + size[0] / 2.0),
                    float(image_center[1] + size[1] / 2.0),
                ]
                output[frame_num] = {1: {"bbox": interpolated_bbox}}

        return output

   
    # DETECTION + TRACKING
   

    def _get_field_detection_model(self):
        if self.field_detection_model is not None:
            return self.field_detection_model

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Pitch filtering requires ROBOFLOW_API_KEY. Add it to the "
                "project .env file before running detection."
            )

        print("Importing Roboflow inference runtime...", flush=True)
        try:
            from inference import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Pitch filtering requires the 'inference' package. "
                "Install project requirements before running detection."
            ) from exc

        model_id = os.getenv(
            "FIELD_DETECTION_MODEL_ID",
            "football-field-detection-f07vi/14"
        )
        print(
            f"Loading hosted pitch-keypoint model {model_id}...",
            flush=True
        )
        self.field_detection_model = get_model(model_id=model_id, api_key=api_key)
        print("Hosted pitch-keypoint model is ready.", flush=True)
        return self.field_detection_model

    def _is_valid_pitch_polygon(self, polygon, frame_shape):
        if polygon is None:
            return False
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) != 4 or not np.isfinite(points).all():
            return False
        if not cv2.isContourConvex(points):
            return False

        height, width = frame_shape[:2]
        area_fraction = abs(cv2.contourArea(points)) / max(width * height, 1)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        polygon_width = max_x - min_x
        polygon_height = max_y - min_y
        if polygon_width <= 0 or polygon_height <= 0:
            return False
        aspect_ratio = polygon_width / polygon_height
        return (
            self.config.pitch_min_area_fraction <= area_fraction
            <= self.config.pitch_max_area_fraction
            and self.config.pitch_min_aspect_ratio <= aspect_ratio
            <= self.config.pitch_max_aspect_ratio
        )

    def _detect_pitch_polygon(
        self,
        frame,
        frame_num,
        camera_matrix=None
    ):
        self.last_pitch_status = "missing"
        warped_previous = None
        if self.last_good_pitch_polygon is not None and camera_matrix is not None:
            warped_previous = cv2.transform(
                self.last_good_pitch_polygon.reshape(1, -1, 2),
                camera_matrix
            ).reshape(-1, 2)

        field_model = self._get_field_detection_model()
        if (
            frame_num == 0
            or frame_num % self.config.field_inference_interval_frames == 0
        ):
            print(
                f"Requesting hosted pitch keypoints for frame "
                f"{frame_num + 1}...",
                flush=True
            )
        results = field_model.infer(
            frame,
            confidence=self.config.field_inference_confidence
        )
        polygon = None
        if results:
            keypoints = sv.KeyPoints.from_inference(results[0])
            if (
                keypoints.xy is not None
                and keypoints.keypoint_confidence is not None
                and len(keypoints.xy) > 0
                and len(keypoints.keypoint_confidence) > 0
            ):
                frame_points = np.asarray(keypoints.xy[0], dtype=np.float32)
                confidences = np.asarray(
                    keypoints.keypoint_confidence[0],
                    dtype=np.float32
                )
                if (
                    frame_points.ndim == 2
                    and frame_points.shape[1] == 2
                    and len(frame_points) == len(self.FIELD_KEYPOINTS)
                    and len(confidences) == len(self.FIELD_KEYPOINTS)
                ):
                    valid = (
                        (confidences >= self.pitch_keypoint_confidence_threshold)
                        & np.isfinite(frame_points).all(axis=1)
                    )
                    homography = None
                    inlier_ratio = 0.0
                    if int(valid.sum()) >= self.min_pitch_keypoints:
                        homography, inliers = cv2.findHomography(
                            self.FIELD_KEYPOINTS[valid],
                            frame_points[valid],
                            cv2.RANSAC,
                            self.config.pitch_ransac_reprojection_px
                        )
                        if homography is not None and inliers is not None:
                            inlier_count = int(inliers.sum())
                            inlier_ratio = inlier_count / max(int(valid.sum()), 1)
                            if (
                                inlier_count >= self.config.pitch_min_inliers
                                and inlier_ratio >= self.config.pitch_min_inlier_ratio
                            ):
                                polygon = cv2.perspectiveTransform(
                                    self.PITCH_BOUNDARY.reshape(1, -1, 2),
                                    homography
                                ).reshape(-1, 2)
                    if not self._is_valid_pitch_polygon(polygon, frame.shape):
                        polygon = None

                    if polygon is not None and warped_previous is not None:
                        unexplained_shift = float(np.mean(np.linalg.norm(
                            polygon - warped_previous,
                            axis=1
                        )))
                        if unexplained_shift > self.config.pitch_max_unexplained_shift_px:
                            polygon = None
                        else:
                            alpha = self.config.pitch_smoothing_alpha
                            polygon = (
                                alpha * polygon
                                + (1.0 - alpha) * warped_previous
                            ).astype(np.float32)

        if polygon is not None:
            self.last_good_pitch_polygon = polygon.astype(np.float32)
            self.last_good_pitch_frame = frame_num
            self.last_pitch_status = "fresh"
            return self.last_good_pitch_polygon

        if (
            warped_previous is not None
            and frame_num - self.last_good_pitch_frame
            <= self.config.pitch_max_reuse_frames
            and self._is_valid_pitch_polygon(warped_previous, frame.shape)
        ):
            self.last_good_pitch_polygon = warped_previous.astype(np.float32)
            self.last_good_pitch_frame = frame_num
            self.last_pitch_status = "reused"
            return self.last_good_pitch_polygon
        return None

    def _point_is_on_pitch(self, point, pitch_polygon):
        if point is None or pitch_polygon is None:
            return False

        distance = cv2.pointPolygonTest(
            pitch_polygon,
            tuple(float(value) for value in point),
            True
        )
        return distance >= -self.touchline_tolerance

    @staticmethod
    def _extract_ground_point_from_detection(bbox):
        x1, y1, x2, y2 = bbox
        return (float((x1 + x2) / 2.0), float(y2))

    def _filter_player_detections(self, player_detection, pitch_polygon):
        if player_detection is None or player_detection.boxes is None:
            return []

        boxes = player_detection.boxes
        if pitch_polygon is None:
            return []

        valid_indices = []

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().tolist()
            ground_point = self._extract_ground_point_from_detection(bbox)
            if self._point_is_on_pitch(ground_point, pitch_polygon):
                valid_indices.append(i)

        return valid_indices

    def detect_frames(self, frames):
        detections = []
        total_frames = len(frames)
        self.last_good_pitch_polygon = None
        self.last_good_pitch_frame = -1
        self.pitch_status_per_frame = []
        cumulative_transform = np.eye(3, dtype=np.float32)
        for frame_num, frame in enumerate(frames):
            if frame_num == 0:
                print(
                    f"Starting detection and pitch filtering for "
                    f"{total_frames} frames...",
                    flush=True
                )
            elif frame_num % 10 == 0:
                print(
                    f"Detection progress: {frame_num}/{total_frames} frames",
                    flush=True
                )

            if (
                frame_num == 0
                or frame_num % self.config.field_inference_interval_frames == 0
            ):
                print(
                    f"Running player detector on frame "
                    f"{frame_num + 1}/{total_frames}...",
                    flush=True
                )
            player_results = self.model.track(
                source=frame,
                conf=self.config.player_detection_confidence,
                tracker=self.tracker_config,
                persist=True,
                verbose=False
            )
            player_detection = player_results[0]
            if frame_num == 0:
                camera_matrix = np.eye(2, 3, dtype=np.float32)
            else:
                camera_matrix, motion, inlier_count, inlier_ratio = (
                    self._estimate_camera_motion(
                        frames[frame_num - 1],
                        frame,
                        excluded_boxes=(
                            self._previous_detection_boxes
                            if hasattr(self, "_previous_detection_boxes")
                            else ()
                        )
                    )
                )
                step_transform = np.vstack([
                    camera_matrix,
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
                ])
                cumulative_transform = step_transform @ cumulative_transform
                self._update_recovery_state(motion, inlier_count, inlier_ratio)
            pitch_polygon = self._detect_pitch_polygon(
                frame,
                frame_num,
                camera_matrix
            )
            self.pitch_status_per_frame.append(self.last_pitch_status)
            ball_referee_results = self.ball_referee_model.predict(
                source=frame,
                conf=self.config.referee_ball_detection_confidence,
                verbose=False
            )
            ball_referee_detection = ball_referee_results[0]
            valid_player_indices = self._filter_player_detections(
                player_detection,
                pitch_polygon
            )
            if pitch_polygon is None:
                print(
                    f"Frame {frame_num}: pitch polygon missing; "
                    "rejecting player detections until pitch is localized",
                    flush=True
                )
            if player_detection.boxes is not None:
                player_detection.boxes = player_detection.boxes[
                    np.asarray(valid_player_indices, dtype=np.int64)
                ]
            if player_detection.boxes is not None:
                self._previous_detection_boxes = (
                    player_detection.boxes.xyxy.cpu().tolist()
                )
            else:
                self._previous_detection_boxes = []
            if frame_num % 10 == 0 or frame_num == total_frames - 1:
                print(
                    f"Frame {frame_num}: pitch polygon "
                    f"{self.last_pitch_status}",
                    flush=True
                )
            detections.append({
                "players": player_detection,
                "ball_referee": ball_referee_detection
            })
            if frame_num % 10 == 0 or frame_num == total_frames - 1:
                print(
                    f"Detection progress: {frame_num + 1}/{total_frames} "
                    f"frames complete ({len(valid_player_indices)} valid players "
                    f"in the latest frame)",
                    flush=True
                )
        return detections

   
    # CAMERA MOVEMENT COMPUTATION (standalone)
   

    def compute_camera_movement(self, frames):
        """
        Compute per‑frame camera translation (tx, ty) using optical flow.
        Returns list of (tx, ty) for each frame (first frame is (0,0)).
        """
        if not frames:
            self.camera_movement_per_frame = []
            self.camera_transform_per_frame = []
            self.camera_inlier_ratio_per_frame = []
            return []

        movement = [(0.0, 0.0)]  # first frame has no movement
        motion_magnitudes = [0.0]
        cumulative_transform = np.eye(3, dtype=np.float32)
        transforms = [cumulative_transform.copy()]
        inlier_ratios = [1.0]

        previous_frame = frames[0]
        for i in range(1, len(frames)):
            matrix, camera_motion, _, inlier_ratio = self._estimate_camera_motion(
                previous_frame,
                frames[i]
            )
            motion_magnitudes.append(camera_motion)
            tx = float(matrix[0, 2])
            ty = float(matrix[1, 2])
            movement.append((tx, ty))
            step_transform = np.vstack(
                [matrix, np.array([[0.0, 0.0, 1.0]], dtype=np.float32)]
            )
            cumulative_transform = step_transform @ cumulative_transform
            transforms.append(cumulative_transform.copy())
            inlier_ratios.append(inlier_ratio)
            previous_frame = frames[i]

        self.camera_movement_per_frame = movement
        self.camera_transform_per_frame = transforms
        self.camera_inlier_ratio_per_frame = inlier_ratios
        self.camera_motion_per_frame = motion_magnitudes
        return movement

   
    # BASIC GEOMETRY HELPERS
   

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

   
    # CAMERA MOTION ESTIMATION
   

    def _estimate_camera_motion(
        self,
        previous_frame,
        current_frame,
        excluded_boxes=None
    ):
        height, width = previous_frame.shape[:2]
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        feature_mask = np.full((height, width), 255, dtype=np.uint8)
        for bbox in excluded_boxes or ():
            x1, y1, x2, y2 = map(int, bbox)
            padding = self.config.camera_mask_padding_px
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            cv2.rectangle(feature_mask, (x1, y1), (x2, y2), 0, cv2.FILLED)
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.gmc_max_corners,
            qualityLevel=self.config.camera_feature_quality_level,
            minDistance=self.config.camera_feature_min_distance_px,
            blockSize=self.config.camera_feature_block_size,
            mask=feature_mask
        )
        if previous_points is None:
            return np.eye(2, 3, dtype=np.float32), 0.0, 0, 0.0
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(
                self.config.camera_flow_window_px,
                self.config.camera_flow_window_px
            ),
            maxLevel=self.config.camera_flow_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.camera_flow_max_iterations,
                self.config.camera_flow_epsilon
            )
        )
        if current_points is None:
            return np.eye(2, 3, dtype=np.float32), 0.0, 0, 0.0
        status = status.reshape(-1)
        good_previous = previous_points.reshape(-1, 2)[status == 1]
        good_current = current_points.reshape(-1, 2)[status == 1]
        if len(good_previous) < self.gmc_min_points:
            return (
                np.eye(2, 3, dtype=np.float32),
                0.0,
                len(good_previous),
                0.0
            )
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            good_previous,
            good_current,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                self.config.camera_affine_ransac_reprojection_px
            ),
            maxIters=self.config.camera_affine_ransac_max_iterations,
            confidence=self.config.camera_affine_ransac_confidence
        )
        if matrix is None:
            return (
                np.eye(2, 3, dtype=np.float32),
                0.0,
                len(good_previous),
                0.0
            )
        inlier_ratio = (
            float(inlier_mask.sum()) / len(inlier_mask)
            if inlier_mask is not None and len(inlier_mask)
            else 0.0
        )
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        linear = matrix[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        rotation = float(np.arctan2(linear[1, 0], linear[0, 0]))
        affine_change = max(abs(scale - 1.0), abs(rotation))
        translation = np.sqrt(tx * tx + ty * ty)
        diagonal = np.sqrt(float(width * width) + float(height * height))
        normalized_motion = max(
            translation / max(diagonal, 1.0),
            affine_change
        )
        return (
            matrix.astype(np.float32),
            normalized_motion,
            len(good_previous),
            inlier_ratio
        )

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

   
    # APPEARANCE FEATURE
   

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

   
    # PREDICT GLOBAL TRACK POSITION
   

    def _predict_global_position(
        self,
        state,
        current_frame_num,
        current_camera_transform
    ):
        age = max(0, current_frame_num - int(state["last_frame"]))
        velocity = np.asarray(
            state.get("velocity", np.zeros(2, dtype=np.float32)),
            dtype=np.float32
        )
        decay = self.config.velocity_decay_per_frame
        motion_steps = sum(decay ** step for step in range(1, age + 1))
        displacement = velocity * motion_steps
        distance = float(np.linalg.norm(displacement))
        if distance > self.config.max_extrapolation_px:
            displacement *= self.config.max_extrapolation_px / distance
        state["prediction_velocity"] = velocity * (decay ** age)
        reference_center = (
            np.asarray(state["reference_center"], dtype=np.float32)
            + displacement
        )
        return self._transform_point(
            reference_center,
            current_camera_transform[:2]
        )

    def assign(self, tid, gid):
        """Register a one-to-one BoT-SORT/global-ID mapping."""
        tid = int(tid)
        gid = int(gid)
        mapped_gid = self.tid_to_gid.get(tid)
        if mapped_gid is not None and mapped_gid != gid:
            raise ValueError(
                f"Tracker ID {tid} is already mapped to global ID {mapped_gid}."
            )
        mapped_tid = self.gid_to_tid.get(gid)
        if mapped_tid is not None and mapped_tid != tid:
            raise ValueError(
                f"Global ID {gid} is already mapped to BoT-SORT ID {mapped_tid}."
            )
        self.tid_to_gid[tid] = gid
        self.gid_to_tid[gid] = tid

    def release(self, gid):
        """Release both directions of a global-ID mapping."""
        gid = int(gid)
        tid = self.gid_to_tid.pop(gid, None)
        if tid is not None and self.tid_to_gid.get(tid) == gid:
            del self.tid_to_gid[tid]
        return tid

    def _create_global_id(
        self,
        tracker_id,
        bbox,
        frame,
        frame_num,
        appearance=None
    ):
        if tracker_id in self.tid_to_gid:
            raise ValueError(
                f"Cannot create a new global ID for mapped tracker ID {tracker_id}."
            )
        if appearance is None:
            appearance = self._appearance_feature(frame, bbox)
        image_center = self._bbox_center(bbox)
        camera_transform = self.camera_transform_per_frame[frame_num]
        reference_center = self._transform_point(
            image_center,
            np.linalg.inv(camera_transform)[:2]
        )

        if self.next_global_id > self.config.max_global_ids:
            raise RuntimeError(
                "Global-ID roster is full; refusing to recycle an ID."
            )
        gid = self.next_global_id
        self.next_global_id += 1

        self.global_tracks[gid] = {
            "center": image_center.copy(),
            "reference_center": reference_center.copy(),
            "predicted_center": image_center.copy(),
            "velocity": np.zeros(2, dtype=np.float32),
            "prediction_velocity": np.zeros(2, dtype=np.float32),
            "bbox": list(bbox),
            "appearance": appearance,
            "appearance_anchor": (
                None if appearance is None else appearance.copy()
            ),
            "last_frame": frame_num,
            "last_tracker_id": tracker_id
        }
        self.assign(tracker_id, gid)
        self.tid_last_seen[tracker_id] = frame_num
        return gid

   
    # GLOBAL ID ASSOCIATION
   

    def _associate_global_ids_legacy(self, frames, tracks):
        return self._associate_global_ids(frames, tracks)

        self.next_global_id = 1
        self.global_tracks = {}
        self.tid_to_gid = {}
        self.gid_to_tid = {}
        self.tid_last_seen = {}
        self.camera_transforms = {}
        self.recovery_until = -1
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
                self.camera_movement_per_frame.append((0.0, 0.0))  # first frame
            else:
                camera_matrix, camera_motion, _ = self._estimate_camera_motion(previous_frame, frame)
                tx = float(camera_matrix[0, 2])
                ty = float(camera_matrix[1, 2])
                self.camera_movement_per_frame.append((tx, ty))

            self.camera_transforms[frame_num] = camera_matrix
            current_detection_count = len(frame_tracks)

            major_camera_motion = (camera_motion >= 0.018)
            severe_camera_motion = (camera_motion >= 0.035)

            if major_camera_motion:
                self.recovery_until = max(self.recovery_until, frame_num + self.recovery_duration)
                if self.debug_camera:
                    print(f"Frame {frame_num}: CAMERA TRANSITION (motion={camera_motion:.3f}, detections={current_detection_count}, previous={self.previous_detection_count})")

            recovery_mode = (frame_num <= self.recovery_until)

            detections = []
            for tracker_id, player in frame_tracks.items():
                bbox = player["bbox"]
                detections.append({
                    "tracker_id": int(player.get("tracker_id", tracker_id)),
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
                gid = self.tid_to_gid.get(tid)
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
                existing_gid = self.tid_to_gid.get(tid)
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

                existing_gid = self.tid_to_gid.get(tid)
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
                    if tid not in self.tid_to_gid:
                        self.assign(tid, best_gid)
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
                camera_only_center = state["center"].copy()
                for motion_frame in range(state["last_frame"] + 1, frame_num + 1):
                    motion_matrix = self.camera_transforms.get(motion_frame)
                    if motion_matrix is not None:
                        camera_only_center = self._transform_point(
                            camera_only_center,
                            motion_matrix
                        )
                elapsed_frames = max(1, frame_num - state["last_frame"])
                velocity = (current_center - camera_only_center) / elapsed_frames
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

                if tid not in self.tid_to_gid:
                    self.assign(tid, gid)
                else:
                    if self.tid_to_gid[tid] != gid:
                        raise RuntimeError(
                            f"BoT ID {tid} maps to global ID "
                            f"{self.tid_to_gid[tid]}, not {gid}."
                        )

                new_frame_tracks[gid] = {
                    "bbox": det["bbox"],
                    "tracker_id": tid,
                    "global_id": gid
                }

            tracks["players"][frame_num] = new_frame_tracks

            mode = "CAMERA-RECOVERY" if recovery_mode else "NORMAL"
            print(f"Frame {frame_num}: {mode} | Global IDs = {list(new_frame_tracks.keys())}")

            previous_frame = frame

        return tracks

    # TRACKER FIX: Bijective BoT-SORT/global-ID association with bounded roster.

    def _update_recovery_state(self, motion, inlier_count, inlier_ratio):
        scene_change = (
            inlier_count >= self.gmc_min_points
            and inlier_ratio < self.config.camera_scene_change_inlier_ratio
        )
        motion_event = (
            motion >= self.config.camera_motion_recovery_threshold
            or scene_change
        )
        quiet_event = (
            motion < self.config.camera_motion_recovery_threshold
            and inlier_count >= self.gmc_min_points
            and inlier_ratio >= self.config.camera_minimum_inlier_ratio
        )

        if not self.recovery_active:
            self.recovery_trigger_count = (
                self.recovery_trigger_count + 1 if motion_event else 0
            )
            if self.recovery_trigger_count >= self.config.recovery_enter_frames:
                self.recovery_active = True
                self.recovery_quiet_count = 0
        else:
            self.recovery_quiet_count = (
                self.recovery_quiet_count + 1 if quiet_event else 0
            )
            if self.recovery_quiet_count >= self.config.recovery_exit_frames:
                self.recovery_active = False
                self.recovery_trigger_count = 0
                self.recovery_quiet_count = 0
        return motion_event, scene_change

    def _association_cost(self, detection, state, predicted, age, recovery):
        distance = float(np.linalg.norm(detection["center"] - predicted))
        position_gate = (
            self.config.position_gate_base_px
            + min(
                age * self.config.position_gate_age_speed_px_per_frame,
                self.config.position_gate_max_age_expansion_px
            )
        )
        if recovery:
            position_gate += self.config.recovery_position_gate_bonus_px
        if distance > position_gate:
            return None, "position-gate"

        appearance_distance = max(
            self._appearance_distance(
                detection["appearance"],
                state.get("appearance")
            ),
            self._appearance_distance(
                detection["appearance"],
                state.get(
                    "appearance_anchor",
                    state.get("appearance")
                )
            )
        )
        if appearance_distance > self.config.appearance_reject_limit:
            return None, "appearance-gate"

        position_score = min(distance / max(position_gate, 1.0), 1.0)
        size_score = self._size_distance(detection["bbox"], state["bbox"])
        predicted_bbox = [
            predicted[0] - (state["bbox"][2] - state["bbox"][0]) / 2.0,
            predicted[1] - (state["bbox"][3] - state["bbox"][1]) / 2.0,
            predicted[0] + (state["bbox"][2] - state["bbox"][0]) / 2.0,
            predicted[1] + (state["bbox"][3] - state["bbox"][1]) / 2.0
        ]
        iou_score = 1.0 - self._bbox_iou(
            detection["bbox"],
            predicted_bbox
        )
        appearance_weight = max(
            self.config.appearance_weight_min,
            self.config.appearance_score_weight
            - age * self.config.appearance_weight_decay_per_frame
        )
        position_weight = (
            self.config.position_score_weight
            + self.config.appearance_score_weight
            - appearance_weight
        )
        cost = (
            position_weight * position_score
            + appearance_weight * appearance_distance
            + self.config.size_score_weight * size_score
            + self.config.iou_score_weight * iou_score
        )
        return float(cost), "scored"

    def _associate_global_ids(self, frames, tracks):
        raw_player_frames = tracks["players"]
        frame_count = min(len(raw_player_frames), len(frames))
        self.next_global_id = 1
        self.global_tracks = {}
        self.tid_to_gid = {}
        self.gid_to_tid = {}
        self.tid_last_seen = {}
        self.tentative_tracks = {}
        self.camera_transforms = {}
        self.camera_step_transforms = {}
        self.camera_movement_per_frame = []
        self.camera_transform_per_frame = []
        self.camera_inlier_ratio_per_frame = []
        self.last_good_pitch_polygon = None
        self.last_good_pitch_frame = -1
        self.recovery_active = False
        self.recovery_trigger_count = 0
        self.recovery_quiet_count = 0
        prior_pitch_status = list(
            getattr(self, "pitch_status_per_frame", [])
        )
        self.pitch_status_per_frame = (
            prior_pitch_status
            if len(prior_pitch_status) >= frame_count
            else ["cached"] * frame_count
        )

        cumulative_transform = np.eye(3, dtype=np.float32)
        self.camera_transforms[0] = cumulative_transform.copy()
        self.camera_step_transforms[0] = np.eye(2, 3, dtype=np.float32)
        output_player_frames = [{} for _ in raw_player_frames]
        unassigned_player_frames = [[] for _ in raw_player_frames]
        previous_frame = None
        previous_boxes = []
        diagnostics = []

        for frame_num in range(frame_count):
            frame = frames[frame_num]
            frame_tracks = raw_player_frames[frame_num]
            rejected = {}
            created = []
            released = []
            expired_mappings = []
            conflicts = []

            if previous_frame is not None:
                camera_matrix, camera_motion, inlier_count, inlier_ratio = (
                    self._estimate_camera_motion(
                        previous_frame,
                        frame,
                        excluded_boxes=previous_boxes
                    )
                )
                tx = float(camera_matrix[0, 2])
                ty = float(camera_matrix[1, 2])
                step_transform = np.vstack([
                    camera_matrix,
                    np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
                ])
                cumulative_transform = step_transform @ cumulative_transform
                self.camera_movement_per_frame.append((tx, ty))
                motion_event, scene_change = self._update_recovery_state(
                    camera_motion,
                    inlier_count,
                    inlier_ratio
                )
                if motion_event and self.recovery_active:
                    print(
                        f"Frame {frame_num}: recovery active "
                        f"(motion={camera_motion:.4f}, "
                        f"inliers={inlier_ratio:.3f}, "
                        f"scene_change={scene_change})",
                        flush=True
                    )
            else:
                camera_matrix = np.eye(2, 3, dtype=np.float32)
                inlier_count = 0
                inlier_ratio = 1.0
                camera_motion = 0.0
                self.camera_movement_per_frame.append((0.0, 0.0))
            self.camera_step_transforms[frame_num] = camera_matrix
            self.camera_transforms[frame_num] = cumulative_transform.copy()
            self.camera_transform_per_frame.append(cumulative_transform.copy())
            self.camera_inlier_ratio_per_frame.append(inlier_ratio)

            detections = []
            for tracker_id, player in frame_tracks.items():
                bbox = player["bbox"]
                detections.append({
                    "tracker_id": int(player.get("tracker_id", tracker_id)),
                    "bbox": list(bbox),
                    "center": self._bbox_center(bbox),
                    "appearance": self._appearance_feature(frame, bbox)
                })
            previous_boxes = [det["bbox"] for det in detections]

            current_tids = {det["tracker_id"] for det in detections}
            for tid, gid in list(self.tid_to_gid.items()):
                last_seen = self.tid_last_seen.get(tid, -1)
                if frame_num - last_seen >= self.config.tid_mapping_max_age:
                    self.release(gid)
                    released.append(gid)
                    expired_mappings.append((tid, gid))

            predicted_positions = {}
            track_ages = {}
            for gid, state in self.global_tracks.items():
                age = frame_num - state["last_frame"]
                if age <= 0 or age > self.config.global_max_age:
                    continue
                predicted = self._predict_global_position(
                    state,
                    frame_num,
                    cumulative_transform
                )
                predicted_positions[gid] = predicted
                track_ages[gid] = age
                state["predicted_center"] = predicted

            assignments = {}
            used_gids = set()
            banned_pairs = set()
            force_candidates = []
            force_by_gid = {}
            for det in detections:
                tid = det["tracker_id"]
                gid = self.tid_to_gid.get(tid)
                if gid is None:
                    continue
                state = self.global_tracks.get(gid)
                if state is None or gid not in predicted_positions:
                    self.release(gid)
                    released.append(gid)
                    continue
                age = track_ages[gid]
                position_error = float(np.linalg.norm(
                    det["center"] - predicted_positions[gid]
                ))
                position_limit = (
                    self.config.forced_position_base_px
                    + self.config.forced_max_speed_px_per_frame * age
                )
                appearance_error = max(
                    self._appearance_distance(
                        det["appearance"],
                        state.get("appearance")
                    ),
                    self._appearance_distance(
                        det["appearance"],
                        state.get(
                            "appearance_anchor",
                            state.get("appearance")
                        )
                    )
                )
                if position_error > position_limit:
                    rejected["forced-position"] = rejected.get("forced-position", 0) + 1
                    banned_pairs.add((tid, gid))
                    self.release(gid)
                    released.append(gid)
                    continue
                if appearance_error > self.config.forced_appearance_limit:
                    rejected["forced-appearance"] = rejected.get("forced-appearance", 0) + 1
                    banned_pairs.add((tid, gid))
                    self.release(gid)
                    released.append(gid)
                    continue
                force_candidates.append((
                    position_error + appearance_error * position_limit,
                    tid,
                    gid
                ))

            for score, tid, gid in force_candidates:
                force_by_gid.setdefault(gid, []).append((score, tid))
            forced_winners = {}
            for gid, candidates in force_by_gid.items():
                candidates.sort()
                forced_winners[gid] = candidates[0][1]
                for _, loser_tid in candidates[1:]:
                    conflicts.append((gid, candidates[0][1], loser_tid))
                    rejected["mapping-conflict"] = (
                        rejected.get("mapping-conflict", 0) + 1
                    )
                    if self.config.debug_invariants:
                        print(
                            f"Frame {frame_num}: mapping conflict gid={gid}; "
                            f"keeping tid={candidates[0][1]}, "
                            f"rematching tid={loser_tid}",
                            flush=True
                        )

            for score, tid, gid in sorted(force_candidates):
                if forced_winners.get(gid) != tid:
                    continue
                if tid in assignments or gid in used_gids:
                    continue
                assignments[tid] = gid
                used_gids.add(gid)

            unmatched = [
                det for det in detections
                if det["tracker_id"] not in assignments
            ]
            eligible_gids = [
                gid for gid in predicted_positions
                if gid not in used_gids
            ]
            eligible_gids.sort()
            invalid_cost = 1.0e6
            costs = np.full(
                (len(unmatched), len(eligible_gids)),
                invalid_cost,
                dtype=np.float64
            )
            for row, det in enumerate(unmatched):
                tid = det["tracker_id"]
                for column, gid in enumerate(eligible_gids):
                    if (tid, gid) in banned_pairs:
                        continue
                    cost, reason = self._association_cost(
                        det,
                        self.global_tracks[gid],
                        predicted_positions[gid],
                        track_ages[gid],
                        self.recovery_active
                    )
                    if cost is None:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue
                    costs[row, column] = cost

            if len(unmatched):
                dummy_cost = self.config.match_acceptance_cost + 1.0e-6
                assignment_costs = np.concatenate(
                    [
                        costs,
                        np.full(
                            (len(unmatched), len(unmatched)),
                            dummy_cost,
                            dtype=np.float64
                        )
                    ],
                    axis=1
                )
                row_indices, column_indices = linear_sum_assignment(
                    assignment_costs
                )
                for row, column in zip(row_indices, column_indices):
                    if column >= len(eligible_gids):
                        continue
                    cost = float(costs[row, column])
                    det = unmatched[row]
                    finite_costs = np.sort(costs[row][costs[row] < invalid_cost])
                    if cost >= self.config.match_acceptance_cost:
                        rejected["cost-threshold"] = (
                            rejected.get("cost-threshold", 0) + 1
                        )
                        continue
                    if (
                        len(finite_costs) > 1
                        and finite_costs[1] - finite_costs[0]
                        < self.config.match_ambiguity_margin
                    ):
                        rejected["ambiguous"] = rejected.get("ambiguous", 0) + 1
                        continue
                    gid = eligible_gids[column]
                    finite_column_costs = np.sort(
                        costs[:, column][costs[:, column] < invalid_cost]
                    )
                    if (
                        len(finite_column_costs) > 1
                        and finite_column_costs[1] - finite_column_costs[0]
                        < self.config.match_ambiguity_margin
                    ):
                        rejected["ambiguous-global-id"] = (
                            rejected.get("ambiguous-global-id", 0) + 1
                        )
                        continue
                    tid = det["tracker_id"]
                    previous_owner = self.gid_to_tid.get(gid)
                    if previous_owner is not None and previous_owner != tid:
                        self.release(gid)
                        self.tid_last_seen.pop(previous_owner, None)
                    self.assign(tid, gid)
                    assignments[tid] = gid
                    used_gids.add(gid)

            for tid in current_tids:
                if tid not in assignments and tid in self.tid_to_gid:
                    released_gid = self.tid_to_gid[tid]
                    self.release(released_gid)
                    released.append(released_gid)

            new_ids_this_frame = 0
            for det in detections:
                tid = det["tracker_id"]
                gid = assignments.get(tid)
                if gid is not None:
                    self.tid_last_seen[tid] = frame_num
                    tentative = self.tentative_tracks.pop(tid, None)
                    if tentative is not None:
                        for pending in tentative["pending"]:
                            pending_frame = pending["frame_num"]
                            if pending in unassigned_player_frames[pending_frame]:
                                unassigned_player_frames[pending_frame].remove(
                                    pending
                                )
                            if pending_frame == frame_num:
                                continue
                            if gid not in output_player_frames[pending_frame]:
                                output_player_frames[pending_frame][gid] = {
                                    "bbox": pending["bbox"],
                                    "tracker_id": tid,
                                    "global_id": gid
                                }
                    continue

                tentative = self.tentative_tracks.get(tid)
                if (
                    tentative is None
                    or frame_num - tentative["last_frame"]
                    > self.config.tentative_max_gap
                ):
                    tentative = {"hits": 0, "last_frame": frame_num, "pending": []}
                    self.tentative_tracks[tid] = tentative
                tentative["hits"] += 1
                tentative["last_frame"] = frame_num
                pending = {
                    "frame_num": frame_num,
                    "bbox": list(det["bbox"]),
                    "tracker_id": tid
                }
                tentative["pending"].append(pending)
                unassigned_player_frames[frame_num].append(pending)

                if (
                    self.recovery_active
                    or tentative["hits"] < self.config.min_confirm_frames
                    or new_ids_this_frame >= self.config.max_new_global_ids_per_frame
                ):
                    continue

                if self.next_global_id > self.config.max_global_ids:
                    rejected["roster-full"] = (
                        rejected.get("roster-full", 0) + 1
                    )
                    continue

                gid = self._create_global_id(
                    tid,
                    det["bbox"],
                    frame,
                    frame_num,
                    det["appearance"]
                )
                assignments[tid] = gid
                used_gids.add(gid)
                self.tid_last_seen[tid] = frame_num
                created.append(gid)
                new_ids_this_frame += 1
                for pending in tentative["pending"]:
                    pending_frame = pending["frame_num"]
                    if pending in unassigned_player_frames[pending_frame]:
                        unassigned_player_frames[pending_frame].remove(pending)
                    if pending_frame == frame_num:
                        continue
                    if gid not in output_player_frames[pending_frame]:
                        output_player_frames[pending_frame][gid] = {
                            "bbox": pending["bbox"],
                            "tracker_id": pending["tracker_id"],
                            "global_id": gid
                        }
                self.tentative_tracks.pop(tid, None)

            for det in detections:
                tid = det["tracker_id"]
                gid = assignments.get(tid)
                if gid is None:
                    continue
                if gid not in self.global_tracks:
                    continue
                state = self.global_tracks[gid]
                reference_center = self._transform_point(
                    det["center"],
                    np.linalg.inv(cumulative_transform)[:2]
                )
                old_reference_center = np.asarray(
                    state["reference_center"],
                    dtype=np.float32
                )
                observed_velocity = (
                    reference_center - old_reference_center
                )
                speed = float(np.linalg.norm(observed_velocity))
                if speed > self.config.max_velocity_px_per_frame:
                    observed_velocity *= (
                        self.config.max_velocity_px_per_frame / speed
                    )
                alpha = self.config.velocity_ema_alpha
                state["velocity"] = (
                    alpha * observed_velocity
                    + (1.0 - alpha)
                    * np.asarray(state["velocity"], dtype=np.float32)
                )
                state["reference_center"] = reference_center.copy()
                state["center"] = det["center"].copy()
                state["predicted_center"] = det["center"].copy()
                state["bbox"] = list(det["bbox"])
                state["last_frame"] = frame_num
                state["last_tracker_id"] = tid
                self.tid_last_seen[tid] = frame_num
                anchor = state.get(
                    "appearance_anchor",
                    state.get("appearance")
                )
                appearance_error = self._appearance_distance(
                    det["appearance"],
                    anchor
                )
                if (
                    det["appearance"] is not None
                    and appearance_error
                    <= self.config.appearance_update_max_distance
                ):
                    if state.get("appearance") is None:
                        state["appearance"] = det["appearance"]
                    else:
                        updated = (
                            0.90 * state["appearance"]
                            + 0.10 * det["appearance"]
                        )
                        norm = np.linalg.norm(updated)
                        if norm > 1e-6:
                            state["appearance"] = (updated / norm).astype(
                                np.float32
                            )
                if gid in output_player_frames[frame_num]:
                    message = (
                        f"Duplicate global ID {gid} in frame {frame_num}."
                    )
                    if self.config.debug_invariants:
                        raise AssertionError(message)
                    raise RuntimeError(message)
                output_player_frames[frame_num][gid] = {
                    "bbox": list(det["bbox"]),
                    "tracker_id": tid,
                    "global_id": gid
                }

            expired_tentatives = [
                tid for tid, state in self.tentative_tracks.items()
                if frame_num - state["last_frame"]
                > self.config.tentative_max_gap
            ]
            for tid in expired_tentatives:
                del self.tentative_tracks[tid]

            active_count = sum(
                frame_num - state["last_frame"] <= self.config.global_max_age
                for state in self.global_tracks.values()
            )
            if self.config.debug_invariants:
                if len(self.global_tracks) > self.config.max_global_ids:
                    raise AssertionError(
                        "Global-ID roster exceeds max_global_ids."
                    )
                if active_count > self.config.max_global_ids:
                    raise AssertionError(
                        "Active global-ID count exceeds max_global_ids."
                    )
                if len(self.tid_to_gid) != len(self.gid_to_tid):
                    raise AssertionError("ID mapping is not bijective.")
                for tid, gid in self.tid_to_gid.items():
                    if self.gid_to_tid.get(gid) != tid:
                        raise AssertionError("ID mapping is not bijective.")
                if len(output_player_frames[frame_num]) != len(
                    set(output_player_frames[frame_num])
                ):
                    raise AssertionError("Duplicate global ID in frame output.")

            polygon_status = self.pitch_status_per_frame[frame_num]
            diagnostics.append({
                "frame": frame_num,
                "active": active_count,
                "created": created,
                "released": released,
                "mapping_expired": expired_mappings,
                "conflicts": conflicts,
                "rejected": rejected,
                "polygon": polygon_status
            })
            rejection_log = (
                ",".join(f"{key}:{value}" for key, value in sorted(rejected.items()))
                or "none"
            )
            print(
                f"Frame {frame_num}: active={active_count}/"
                f"{self.config.max_global_ids} created={created} "
                f"released={released} conflicts={len(conflicts)} "
                f"rejected={rejection_log} polygon={polygon_status}",
                flush=True
            )
            previous_frame = frame

        tracks["players"] = output_player_frames
        metadata = tracks.get("_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["polygon_status_per_frame"] = list(
            self.pitch_status_per_frame[:frame_count]
        )
        metadata["unassigned_players_per_frame"] = (
            unassigned_player_frames
        )
        tracks["_metadata"] = metadata
        self.association_diagnostics = diagnostics
        return tracks

    # ADD POSITIONS TO TRACKS
   

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

    def remove_players_overlapping_referees(self, tracks, iou_threshold=0.5):
        """Remove player detections that duplicate a referee detection."""
        players = tracks.get("players")
        referees = tracks.get("referees")
        if not isinstance(players, list) or not isinstance(referees, list):
            raise ValueError("Tracks must contain player and referee frame lists.")
        if len(players) != len(referees):
            raise ValueError("Player and referee track frame counts must match.")

        removed_count = 0
        for player_frame, referee_frame in zip(players, referees):
            referee_boxes = [
                referee.get("bbox")
                for referee in referee_frame.values()
                if referee.get("bbox") is not None
            ]
            duplicate_player_ids = [
                player_id
                for player_id, player in player_frame.items()
                if any(
                    self._bbox_iou(player["bbox"], referee_bbox) >= iou_threshold
                    for referee_bbox in referee_boxes
                )
            ]
            for player_id in duplicate_player_ids:
                del player_frame[player_id]
            removed_count += len(duplicate_player_ids)

        if removed_count:
            print(
                f"Removed {removed_count} player detections overlapping "
                "referee detections."
            )
        return tracks

   
    # DRAW CAMERA MOVEMENT
   

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

   
    # GET TRACKS
   

    def get_objects_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        allow_recompute=True
    ):
        if read_from_stub and not allow_recompute and stub_path is None:
            raise ValueError(
                "stub_path is required when recomputation is disabled."
            )

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            track_types = ("players", "referees", "ball")
            if (
                isinstance(tracks, dict)
                and all(
                    isinstance(tracks.get(track_type), list)
                    and len(tracks[track_type]) == len(frames)
                    for track_type in track_types
                )
            ):
                metadata = tracks.get("_metadata", {})
                cached_pitch_status = (
                    metadata.get("polygon_status_per_frame")
                    if isinstance(metadata, dict)
                    else None
                )
                self.pitch_status_per_frame = (
                    list(cached_pitch_status)
                    if isinstance(cached_pitch_status, list)
                    and len(cached_pitch_status) == len(frames)
                    else ["cached"] * len(frames)
                )
                print(
                    f"Re-associating cached detections from {stub_path}; "
                    "detector and hosted pitch-model inference are skipped.",
                    flush=True
                )
                tracks = self._associate_global_ids(frames, tracks)
                print(
                    f"Replayed cached tracks from {stub_path}; original stub "
                    "was not modified.",
                    flush=True
                )
                return tracks

            if not allow_recompute:
                raise RuntimeError(
                    f"Track cache {stub_path} is stale or incompatible with "
                    "the current video. Refusing to run expensive inference; "
                    "set allow_recompute=True to rebuild it explicitly."
                )

            print(
                "Existing track stub is stale or lacks current pitch filtering. "
                "Regenerating tracks..."
            )
        elif read_from_stub and not allow_recompute:
            raise FileNotFoundError(
                f"Track cache {stub_path} was not found. Refusing to run "
                "expensive inference; set allow_recompute=True to rebuild it "
                "explicitly."
            )

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
        tracks["_metadata"] = {
            "pitch_filter_version": self.PITCH_FILTER_VERSION
        }

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"Tracks saved to {stub_path}")

        return tracks

   
    # DRAW ELLIPSE
   

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
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            text = str(track_id)
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            text_x = x_center - text_size[0] // 2
            text_y = y1_rect + (rect_height + text_size[1]) // 2
            cv2.putText(
                frame,
                text,
                (int(text_x), int(text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_ball_marker(frame, bbox):
        center = Tracker._bbox_center(bbox)
        radius = max(
            5,
            min(10, int(round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0)))
        )
        point = (int(round(center[0])), int(round(center[1])))
        cv2.circle(frame, point, radius + 2, (15, 20, 20), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(
            frame,
            point,
            max(1, radius // 3),
            (235, 255, 235),
            cv2.FILLED,
            cv2.LINE_AA
        )
        return frame

    def draw_team_ball_control(
        self,
        frame,
        frame_num,
        team_ball_control,
        team_labels=None,
        team_colors=None,
        team_pass_stats=None
    ):
        team_labels = team_labels or {1: "Team 1", 2: "Team 2"}
        team_colors = team_colors or {}
        frame_height, frame_width = frame.shape[:2]
        panel_width = min(420, frame_width - 24)
        panel_height = min(210, frame_height - 24)
        panel_x1 = frame_width - panel_width - 12
        panel_y1 = frame_height - panel_height - 12
        panel_x2 = frame_width - 12
        panel_y2 = frame_height - 12

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (20, 25, 30),
            cv2.FILLED
        )
        alpha = 0.84
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

        label_colors = {
            team_id: tuple(
                int(max(0, min(255, round(channel))))
                for channel in team_colors.get(team_id, (255, 255, 255))
            )
            for team_id in (1, 2)
        }

        font_scale = max(0.4, min(0.75, frame_height / 960))
        thickness = max(1, round(font_scale * 2))
        padding = max(8, round(frame_height * 0.015))
        text_x = panel_x1 + padding
        line_height = cv2.getTextSize(
            "TEAM", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][1]
        heading_y = panel_y1 + padding + line_height

        possession_heading = "POSSESSION"
        cv2.putText(
            frame,
            possession_heading,
            (text_x, heading_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        row_gap = max(4, round(font_scale * 10))
        row_y = heading_y + line_height + row_gap
        for team_id, percentage in ((1, team1_percent), (2, team2_percent)):
            label = str(team_labels.get(team_id, f"Team {team_id}")).upper()
            percentage_text = f"{percentage:5.1f}%"
            percentage_width = cv2.getTextSize(
                percentage_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0][0]
            label_max_width = panel_width - 2 * padding - percentage_width - padding
            while label and cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0][0] > label_max_width:
                label = label[:-1]
            if label != str(team_labels.get(team_id, f"Team {team_id}")).upper():
                label = label.rstrip() + "..."
            cv2.putText(
                frame,
                label,
                (text_x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_colors[team_id],
                thickness,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                percentage_text,
                (panel_x2 - padding - percentage_width, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_colors[team_id],
                thickness,
                cv2.LINE_AA
            )
            row_y += line_height + row_gap

        divider_y = row_y + max(2, padding // 2)
        cv2.line(
            frame,
            (text_x, divider_y),
            (panel_x2 - padding, divider_y),
            (90, 100, 108),
            1,
            cv2.LINE_AA
        )
        passes_heading_y = divider_y + line_height + row_gap
        passes_heading = "PASSES  (COMPLETED / ATTEMPTED  -  INCOMPLETE)"
        passes_heading_scale = max(0.35, font_scale * 0.78)
        passes_heading_width = cv2.getTextSize(
            passes_heading,
            cv2.FONT_HERSHEY_SIMPLEX,
            passes_heading_scale,
            1
        )[0][0]
        available_heading_width = panel_width - 2 * padding
        if passes_heading_width > available_heading_width:
            passes_heading_scale *= available_heading_width / passes_heading_width
        cv2.putText(
            frame,
            passes_heading,
            (text_x, passes_heading_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            passes_heading_scale,
            (230, 235, 240),
            1,
            cv2.LINE_AA
        )

        pass_row_y = passes_heading_y + line_height + row_gap
        for team_id in (1, 2):
            team_counts = (team_pass_stats or {}).get(str(team_id), {})
            completed = int(team_counts.get("passes_completed", 0))
            attempted = int(team_counts.get("passes_attempted", 0))
            incomplete = int(team_counts.get("passes_incomplete", 0))
            count_text = f"{completed}/{attempted}  ({incomplete} INC)"
            count_width = cv2.getTextSize(
                count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0][0]
            label = str(team_labels.get(team_id, f"Team {team_id}")).upper()
            label_max_width = panel_width - 2 * padding - count_width - padding
            while label and cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0][0] > label_max_width:
                label = label[:-1]
            if label != str(team_labels.get(team_id, f"Team {team_id}")).upper():
                label = label.rstrip() + "..."
            cv2.putText(
                frame,
                label,
                (text_x, pass_row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_colors[team_id],
                thickness,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                count_text,
                (panel_x2 - padding - count_width, pass_row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_colors[team_id],
                thickness,
                cv2.LINE_AA
            )
            pass_row_y += line_height + row_gap

        return frame

    def draw_annotations(
        self,
        video_frames,
        tracks,
        team_ball_control,
        team_labels=None,
        team_colors=None,
        pass_stats_timeline=None
    ):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get(
                    "team_color",
                    self.UNASSIGNED_PLAYER_COLOR
                )
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))

            unassigned_frames = tracks.get("_metadata", {}).get(
                "unassigned_players_per_frame", []
            )
            if frame_num < len(unassigned_frames):
                for player in unassigned_frames[frame_num]:
                    frame = self.draw_ellipse(
                        frame,
                        player["bbox"],
                        self.UNASSIGNED_PLAYER_COLOR
                    )

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame, referee["bbox"], self.REFEREE_COLOR
                )

            for _, ball in ball_dict.items():
                frame = self.draw_ball_marker(frame, ball["bbox"])

            frame = self.draw_team_ball_control(
                frame,
                frame_num,
                team_ball_control,
                team_labels,
                team_colors,
                (
                    pass_stats_timeline[frame_num]
                    if pass_stats_timeline and frame_num < len(pass_stats_timeline)
                    else None
                )
            )

            output_video_frames.append(frame)
        return output_video_frames