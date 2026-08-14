from ultralytics import YOLO
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

        self.ball_referee_model = YOLO(
            r"C:\football-yolo\models\ball_referee.pt"
        )

        # BoT-SORT + ReID configuration
        self.tracker_config = r"C:\football-yolo\track_player\botsort_reid.yaml"

        self.draw = sv.EllipseAnnotator()


    def detect_frames(self, frames):

        detections = []

        # -------------------------------------------------
        # IMPORTANT:
        # Process ONE frame at a time.
        #
        # Do not pass frames[i:i+batch_size] to
        # model.track() because persist=True needs the
        # tracker state to continue from one frame to
        # the next.
        # -------------------------------------------------

        for frame_num, frame in enumerate(frames):

            # -------------------------------------------------
            # YOLOv26 - PLAYERS + BoT-SORT + ReID
            # -------------------------------------------------

            player_results = self.model.track(
                source=frame,
                conf=0.1,
                tracker=self.tracker_config,
                persist=True,
                verbose=False
            )

            player_detection = player_results[0]


            # -------------------------------------------------
            # YOLO11 - BALL + REFEREE
            # -------------------------------------------------

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


    def get_objects_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

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
            # PLAYER MODEL - YOLOv26 + BoT-SORT + ReID
            # -------------------------------------------------

            player_detection = detection["players"]

            tracks["players"].append({})


            if (
                player_detection.boxes is not None
                and player_detection.boxes.id is not None
            ):

                boxes = player_detection.boxes


                for i in range(len(boxes)):

                    bbox = boxes.xyxy[i].cpu().tolist()

                    track_id = int(
                        boxes.id[i].cpu().item()
                    )


                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox
                    }


            # -------------------------------------------------
            # BALL + REFEREE MODEL - YOLO11
            # -------------------------------------------------

            ball_referee_detection = detection["ball_referee"]


            ball_referee_supervision = sv.Detections.from_ultralytics(
                ball_referee_detection
            )


            referee_indices = []
            ball_indices = []


            for i, class_id in enumerate(
                ball_referee_supervision.class_id
            ):

                class_name = ball_referee_detection.names[class_id]


                if class_name == "Referee":
                    referee_indices.append(i)

                elif class_name == "Ball":
                    ball_indices.append(i)


            # -------------------------------------------------
            # REFEREES
            # -------------------------------------------------

            tracks["referees"].append({})


            if len(referee_indices) > 0:

                for i in referee_indices:

                    bbox = ball_referee_supervision.xyxy[i].tolist()


                    tracks["referees"][frame_num][i] = {
                        "bbox": bbox
                    }


            # -------------------------------------------------
            # BALL
            # -------------------------------------------------

            tracks["ball"].append({})


            if len(ball_indices) > 0:

                # Keep only the highest-confidence ball
                best_ball_index = max(
                    ball_indices,
                    key=lambda i:
                    ball_referee_supervision.confidence[i]
                )


                bbox = ball_referee_supervision.xyxy[
                    best_ball_index
                ].tolist()


                tracks["ball"][frame_num][1] = {
                    "bbox": bbox
                }


            print(
                f"Frame {frame_num}: "
                f"{len(tracks['players'][frame_num])} players, "
                f"{len(tracks['referees'][frame_num])} referees, "
                f"{len(tracks['ball'][frame_num])} ball"
            )


        # -------------------------------------------------
        # SAVE TRACKS
        # -------------------------------------------------

        if stub_path is not None:

            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)


            print(
                f"Tracks saved to {stub_path}"
            )


        return tracks


    def draw_ellipse(
        self,
        frame,
        bbox,
        color,
        track_id=None
    ):

        y2 = int(bbox[3])


        x_center, _ = get_center_of_bbox(bbox)

        width = get_bbox_width(bbox)


        # -------------------------------------------------
        # PLAYER / REFEREE ELLIPSE
        # -------------------------------------------------

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(
                int(width),
                int(0.35 * width)
            ),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )


        # -------------------------------------------------
        # ID LABEL
        # -------------------------------------------------

        if track_id is not None:

            rect_width = 40
            rect_height = 20


            x1_rect = (
                x_center
                - rect_width // 2
            )


            x2_rect = (
                x_center
                + rect_width // 2
            )


            y1_rect = y2 + 5

            y2_rect = (
                y1_rect
                + rect_height
            )


            cv2.rectangle(
                frame,
                (
                    int(x1_rect),
                    int(y1_rect)
                ),
                (
                    int(x2_rect),
                    int(y2_rect)
                ),
                color,
                cv2.FILLED
            )


            # -------------------------------------------------
            # CENTER ID TEXT
            # -------------------------------------------------

            text = str(track_id)


            text_size = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]


            text_x = (
                x_center
                - text_size[0] // 2
            )


            text_y = (
                y1_rect
                + (
                    rect_height
                    + text_size[1]
                ) // 2
            )


            cv2.putText(
                frame,
                text,
                (
                    int(text_x),
                    int(text_y)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )


        return frame


    def draw_annotations(
        self,
        video_frames,
        tracks
    ):

        output_video_frames = []


        for frame_num, frame in enumerate(
            video_frames
        ):

            frame = frame.copy()


            player_dict = tracks["players"][frame_num]

            ball_dict = tracks["ball"][frame_num]

            referee_dict = tracks["referees"][frame_num]


            # -------------------------------------------------
            # PLAYERS
            # -------------------------------------------------

            for track_id, player in player_dict.items():

                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    (0, 0, 255),
                    track_id
                )


            # -------------------------------------------------
            # REFEREES
            # -------------------------------------------------

            for _, referee in referee_dict.items():

                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    (0, 255, 255)
                )


            # -------------------------------------------------
            # BALL
            # -------------------------------------------------

            for _, ball in ball_dict.items():

                bbox = ball["bbox"]


                x_center, y_center = (
                    get_center_of_bbox(bbox)
                )


                cv2.circle(
                    frame,
                    (
                        int(x_center),
                        int(y_center)
                    ),
                    10,
                    (0, 255, 255),
                    -1
                )


            output_video_frames.append(frame)


        return output_video_frames