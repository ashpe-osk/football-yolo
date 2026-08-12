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
        self.tracker = ByteTrack()  
        self.draw = sv.EllipseAnnotator()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
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
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inv["player"]

            # Track objects
            det_w_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize dictionaries for this frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

       
            for i in range(len(det_w_tracks)):
                bbox = det_w_tracks.xyxy[i].tolist()
                class_id = det_w_tracks.class_id[i]
                track_id = det_w_tracks.tracker_id[i]
                
                if cls_names[class_id] == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_names[class_id] == "referee":  # FIXED: "referee" not "referees"
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Process ball detections (ball is not tracked, just detected)
            for i in range(len(detection_supervision)):
                if cls_names[detection_supervision.class_id[i]] == "ball":
                    bbox = detection_supervision.xyxy[i].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            print(f"Frame {frame_num}: {len(det_w_tracks)} tracked objects")

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"Tracks saved to {stub_path}")
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, _= get_center_of_bbox(bbox)
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
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)

            output_video_frames.append(frame)
        return output_video_frames