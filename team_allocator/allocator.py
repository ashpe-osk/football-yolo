from sklearn.cluster import KMeans
import cv2
import numpy as np


class TeamAllocator:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
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

        # ---------------------------------------------------------
        # CENTRAL TORSO REGION
        # ---------------------------------------------------------
        #
        # Ignore:
        # - head
        # - legs
        # - most background
        #
        # Focus on the middle part where the jersey is.
        #

        torso_y1 = int(player_h * 0.20)
        torso_y2 = int(player_h * 0.65)

        torso_x1 = int(player_w * 0.20)
        torso_x2 = int(player_w * 0.80)

        torso = player_image[
            torso_y1:torso_y2,
            torso_x1:torso_x2
        ]

        if torso.size == 0:
            torso = player_image

        # ---------------------------------------------------------
        # REMOVE VERY GREEN PIXELS
        # ---------------------------------------------------------
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

        # If too few useful pixels remain, use the whole torso
        if len(pixels) < 20:
            pixels = torso.reshape(-1, 3)

        # ---------------------------------------------------------
        # KMEANS ON TORSO
        # ---------------------------------------------------------

        if len(pixels) < 2:
            return np.mean(torso.reshape(-1, 3), axis=0)

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        kmeans.fit(pixels)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Count pixels in each cluster
        counts = np.bincount(labels)

        # The dominant cluster is usually the jersey
        player_cluster = np.argmax(counts)

        player_color = centers[player_cluster]

        return player_color

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

        # ---------------------------------------------------------
        # CLUSTER THE PLAYERS INTO TWO TEAMS
        # ---------------------------------------------------------

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            random_state=42
        )

        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        print("\nTeam colors initialized:")
        print(f"Team 1 color: {self.team_colors[1]}")
        print(f"Team 2 color: {self.team_colors[2]}")

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine the team of a player using their Global ID.

        Multiple observations are stored for each Global ID instead
        of trusting one single frame.
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

        # ---------------------------------------------------------
        # STORE COLOR HISTORY FOR THIS GLOBAL PLAYER ID
        # ---------------------------------------------------------

        if player_id not in self.player_color_history:
            self.player_color_history[player_id] = []

        self.player_color_history[player_id].append(player_color)

        # Keep only the most recent observations
        if len(self.player_color_history[player_id]) > 10:
            self.player_color_history[player_id] = \
                self.player_color_history[player_id][-10:]

        # ---------------------------------------------------------
        # USE AVERAGE COLOR FROM MULTIPLE OBSERVATIONS
        # ---------------------------------------------------------

        color_history = np.array(
            self.player_color_history[player_id]
        )

        average_color = np.mean(
            color_history,
            axis=0
        )

        # ---------------------------------------------------------
        # PREDICT TEAM
        # ---------------------------------------------------------

        team_id = self.kmeans.predict(
            average_color.reshape(1, -1)
        )[0]

        team_id = int(team_id) + 1

        # Store current team
        self.player_team_dict[player_id] = team_id

        return team_id