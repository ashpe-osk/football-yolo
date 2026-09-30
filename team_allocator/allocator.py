from sklearn.cluster import KMeans
import cv2
import numpy as np


class TeamAllocator:
    TORSO_Y_START = 0.18
    TORSO_Y_END = 0.50
    TORSO_X_START = 0.35
    TORSO_X_END = 0.65
    MIN_USABLE_PIXELS = 20
    MIN_TRACK_SAMPLES = 1

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}
        self.player_color_history = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        """
        Cluster the pixels in an image into two color groups.
        """

        image_2d = image.reshape((-1, 3))

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant jersey color from the central torso
        region of the player.

        This avoids using the entire bounding box because it can
        contain grass, legs, background, etc.
        """

        x1, y1, x2, y2 = map(int, bbox)

        # Clamp coordinates to the frame
        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # Invalid bounding box
        if x2 <= x1 or y2 <= y1:
            return np.array([0.0, 0.0, 0.0])

        player_image = frame[y1:y2, x1:x2]

        if player_image.size == 0:
            return np.array([0.0, 0.0, 0.0])

        player_h, player_w = player_image.shape[:2]

        
        # CENTRAL TORSO REGION
        
        #
        # Ignore:
        # - head
        # - legs
        # - most background
        #
        # Focus on the middle part where the jersey is.
        #

        torso_y1 = int(player_h * self.TORSO_Y_START)
        torso_y2 = int(player_h * self.TORSO_Y_END)
        torso_x1 = int(player_w * self.TORSO_X_START)
        torso_x2 = int(player_w * self.TORSO_X_END)

        torso = player_image[
            torso_y1:torso_y2,
            torso_x1:torso_x2
        ]

        if torso.size == 0:
            torso = player_image

        
        # REMOVE VERY GREEN PIXELS
        
        #
        # The pitch is green, so remove pixels that are strongly
        # green. This prevents grass from becoming the "jersey".
        #

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 30])
        upper_green = np.array([95, 255, 255])

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        non_green_mask = cv2.bitwise_not(green_mask)

        pixels = torso[non_green_mask > 0]

        if len(pixels) < self.MIN_USABLE_PIXELS:
            pixels = torso.reshape(-1, 3)

        # Median color is less affected than per-crop KMeans by skin, stripes,
        # and a few background pixels leaking into small player boxes.
        return np.median(pixels, axis=0).astype(np.float32)

    def allocate_teams(self, frame, player_detections):
        """
        Determine the two team color clusters from the players
        visible in the supplied frame.
        """

        player_colors = []

        for _, player_detection in player_detections.items():

            bbox = player_detection["bbox"]

            player_color = self.get_player_color(
                frame,
                bbox
            )

            player_colors.append(player_color)

        # Need at least two players
        if len(player_colors) < 2:
            print("Not enough players to allocate teams.")
            return

        player_colors = np.array(player_colors)

        
        # CLUSTER THE PLAYERS INTO TWO TEAMS
        

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            random_state=42
        )

        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }

        print("\nTeam colors initialized:")
        print(f"Team 1 color: {self.team_colors[1]}")
        print(f"Team 2 color: {self.team_colors[2]}")

    def get_team_labels(self):
        """Map the detected red and white kits to the known team names."""
        team_scores = {}
        for team_id, color in self.team_colors.items():
            blue, green, red = color
            brightness = max(color) / 255.0
            saturation = (max(color) - min(color)) / 255.0
            red_score = max(0.0, red - (blue + green) / 2.0) / 255.0
            white_score = brightness * (1.0 - saturation)
            team_scores[team_id] = (red_score, white_score)

        red_team = max(team_scores, key=lambda team_id: team_scores[team_id][0])
        white_team = 1 if red_team == 2 else 2
        return {
            red_team: "Kenya (Red)",
            white_team: "Uganda (White)"
        }

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine the team of a player using their Global ID.

        Record the observed team vote for later per-track stabilization.
        """

        if self.kmeans is None:
            raise RuntimeError(
                "Teams have not been allocated. "
                "Call allocate_teams() first."
            )

        player_color = self.get_player_color(
            frame,
            player_bbox
        )

        
        if not np.isfinite(player_color).all() or np.linalg.norm(player_color) <= 1:
            return self.player_team_dict.get(player_id, 0)

        self.player_color_history.setdefault(player_id, []).append(
            player_color.astype(np.float32)
        )
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        self.player_team_history.setdefault(player_id, []).append(team_id)
        self.player_team_dict[player_id] = team_id
        return team_id

    def finalize_player_teams(self):
        """Assign each track from a majority vote of its first five samples."""
        if self.kmeans is None:
            raise RuntimeError(
                "Teams have not been allocated. "
                "Call allocate_teams() first."
            )

        track_colors = {}
        for player_id, samples in self.player_color_history.items():
            if len(samples) < self.MIN_TRACK_SAMPLES:
                continue
            track_colors[player_id] = np.median(
                np.asarray(samples, dtype=np.float32),
                axis=0
            )

        if len(track_colors) < 2:
            raise RuntimeError(
                "At least two sampled player tracks are needed to finalize teams."
            )

        player_ids = list(track_colors)
        aggregated_colors = np.asarray(
            [track_colors[player_id] for player_id in player_ids],
            dtype=np.float32
        )
        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            random_state=42
        )
        kmeans.fit(aggregated_colors)
        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }
        self.player_team_dict = {
            player_id: int(team_id) + 1
            for player_id, team_id in zip(
                player_ids,
                kmeans.labels_
            )
        }

        return self.player_team_dict.copy()