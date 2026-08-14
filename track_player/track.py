from ultralytics import YOLO
from supervision.tracker.byte_tracker.core import ByteTrack
import supervision as sv
import cv2
import pickle
import os 
import sys
sys.path.append("../")
from utils.bbox_utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ball_referee_model = YOLO(r"C:\football-yolo\models\ball_referee.pt") 
        self.tracker = ByteTrack()  
        self.draw = sv.EllipseAnnotator()

    def detect_frames(self, frames):
        batch_size = 1
        detections = []

        for i in range(0, len(frames), batch_size):
            
            player_detections = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1
            )

            ball_referee_detections = self.ball_referee_model.predict(
                frames[i:i+batch_size],
                conf=0.1
            )

            for player_detection, ball_referee_detection in zip(
                player_detections,
                ball_referee_detections
            ):
                detections.append({
                    "players": player_detection,
                    "ball_referee": ball_referee_detection
                })

        return detections
            
    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):
    
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):

            # -------------------------------------------------
            # PLAYER MODEL - YOLOv26 SportsMOT
            # -------------------------------------------------

            player_detection = detection["players"]

            player_supervision = sv.Detections.from_ultralytics(
                player_detection
            )

            # YOLOv26 SportsMOT detects players only,
            # including goalkeepers
            for i in range(len(player_supervision)):
                player_supervision.class_id[i] = 0


            # -------------------------------------------------
            # BALL + REFEREE MODEL - Soccana YOLO11
            # -------------------------------------------------

            ball_referee_detection = detection["ball_referee"]

            ball_referee_supervision = sv.Detections.from_ultralytics(
                ball_referee_detection
            )

            referee_indices = []
            ball_indices = []

            for i, class_id in enumerate(ball_referee_supervision.class_id):

                class_name = ball_referee_detection.names[class_id]

                if class_name == "Referee":
                    referee_indices.append(i)

                elif class_name == "Ball":
                    ball_indices.append(i)


            # -------------------------------------------------
            # PROCESS REFEREES
            # -------------------------------------------------

            if len(referee_indices) > 0:

                referee_supervision = ball_referee_supervision[
                    referee_indices
                ]

                # Give referee class ID 1
                for i in range(len(referee_supervision)):
                    referee_supervision.class_id[i] = 1

            else:
                referee_supervision = sv.Detections.empty()


            # -------------------------------------------------
            # PROCESS BALL
            # -------------------------------------------------

            tracks["ball"].append({})

            if len(ball_indices) > 0:

                # Keep the ball detection
                # Ball is not tracked by ByteTrack

                best_ball_index = max(
                    ball_indices,
                    key=lambda i: ball_referee_supervision.confidence[i]
                )

                bbox = ball_referee_supervision.xyxy[
                    best_ball_index
                ].tolist()

                tracks["ball"][frame_num][1] = {
                    "bbox": bbox
                }


            # -------------------------------------------------
            # COMBINE PLAYERS + REFEREES
            # -------------------------------------------------

            detection_supervision = sv.Detections.merge([
                player_supervision,
                referee_supervision
            ])


            # -------------------------------------------------
            # Track objects
            # -------------------------------------------------

            det_w_tracks = self.tracker.update_with_detections(
                detection_supervision
            )


            # Initialize dictionaries for this frame
            tracks["players"].append({})
            tracks["referees"].append({})


            # -------------------------------------------------
            # SAVE TRACKED OBJECTS
            # -------------------------------------------------

            if det_w_tracks is not None and len(det_w_tracks) > 0:

                for i in range(len(det_w_tracks)):

                    bbox = det_w_tracks.xyxy[i].tolist()
                    class_id = det_w_tracks.class_id[i]
                    track_id = det_w_tracks.tracker_id[i]

                    if class_id == 0:

                        tracks["players"][frame_num][track_id] = {
                            "bbox": bbox
                        }

                    elif class_id == 1:

                        tracks["referees"][frame_num][track_id] = {
                            "bbox": bbox
                        }


            print(
                f"Frame {frame_num}: "
                f"{len(tracks['players'][frame_num])} players, "
                f"{len(tracks['referees'][frame_num])} referees, "
                f"{len(tracks['ball'][frame_num])} ball"
            )


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

            print(f"Tracks saved to {stub_path}")
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):

        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame

    def draw_annotations(self, video_frames, tracks):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]


            # Draw Players 
            for track_id, player in player_dict.items():

                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    (0,0,255),
                    track_id
                )


            # Draw Referees
            for track_id, referee in referee_dict.items():

                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    (0,255,0),
                    track_id
                )


            # Draw Ball
            for ball_id, ball in ball_dict.items():

                bbox = ball["bbox"]

                x_center, y_center = get_center_of_bbox(bbox)

                cv2.circle(
                    frame,
                    (int(x_center), int(y_center)),
                    10,
                    (0,255,255),
                    -1
                )


            output_video_frames.append(frame)

        return output_video_frames