from pathlib import Path
from time import time
from tkinter import filedialog, simpledialog

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class ScoreCR:
    def __init__(self, video_path=None):
        """
        Initialize the ScoreCR scoring tool.
        Loads the video, sets up all required state, and prepares output paths for scoring and properties.
        If no video_path is provided, prompts the user to select a video file.

        Args:
            video_path (str, optional): Path to the video file. If None, opens file dialog.
        """
        self.window = (
            str(video_path) if video_path is not None else filedialog.askopenfilename()
        )
        if self.window:
            self.video_path = Path(self.window)
            self.video = cv2.VideoCapture(self.window)
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.resize_factor = 720 / min(self.height, self.width)
            self.frame = None
            self.colors = {
                "r": (0, 255, 255),
                "l": (255, 255, 0),
                "b": (255, 0, 255),
                "n": (63, 63, 63),
            }

            self.step = 1
            self.key = None
            self.play = False
            self.run = True
            self.stamp = [0, 0]

            self.mouse = None
            self.mouse_start = None
            self.mouse_end = None

            self.view_mode = "bottom"
            self.rois = []
            self.hsv_lower = (0, 0, 0)
            self.hsv_upper = (180, 255, 50)

            self.score_path = self.video_path.with_name(
                f"{self.video_path.stem}_score.csv"
            )
            self.properties_path = self.video_path.with_name(
                f"{self.video_path.stem}_properties.csv"
            )
            self.screenshot_path = self.video_path.with_name(
                f"{self.video_path.stem}_screenshot.png"
            )

            self.score = (
                pd.read_csv(self.score_path)
                if self.score_path.exists()
                else pd.DataFrame(
                    [["n", pd.NA, pd.NA, pd.NA, pd.NA]] * self.frame_count,
                    columns=["label", "l", "r", "b", "n"],
                )
            )
            self.properties = (
                pd.read_csv(self.properties_path)
                if self.properties_path.exists()
                else None
            )

            self.frame_index = (
                self.score[self.score["label"].ne("n")].last_valid_index() or 0
            )
            self.main()
        return

    def main(self):
        """
        Main event loop for the scoring tool.
        Handles frame updates, user input, drawing overlays, and interface refresh.
        Restarts the tool upon closing for continuous workflow.
        """
        self.create_window()
        while self.run:
            self.update_frame()
            self.play_frame()
            self.monitor_key()
            self.draw_circle()
            self.draw_overlay()
            self.draw_peaks()
            self.print_frame()
        self.close()
        ScoreCR()
        return

    def play_frame(self):
        """
        Handles frame advancement in play/fast-forward mode.
        Updates frame index based on elapsed time and user-defined step size.
        Captures keyboard input and checks for window closure.
        """
        if self.play:
            diff = (
                time()
                - self.stamp[0]
                - (self.frame_index - self.stamp[1]) / self.fps / self.step
            )
            self.update_frame_index(self.frame_index + diff * self.fps * self.step)
        self.key = chr(cv2.waitKey(1) & 0xFF)
        try:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                self.run = False
        except cv2.error:
            self.run = False
        return

    def extract_roi(self, frame, rois):
        """
        Extracts a region of interest (ROI) from the frame.
        Supports single circle (bottom mode) or annulus (top mode).

        Args:
            frame (numpy.ndarray): Input frame
            rois (list): List of (center, radius) tuples

        Returns:
            numpy.ndarray: Extracted region
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if len(rois) == 1:
            center, radius = rois[0]
            cv2.circle(mask, center, radius, 255, -1)
            ref_center = center
            ref_radius = radius
        elif len(rois) == 2:
            sorted_rois = sorted(rois, key=lambda x: x[1], reverse=True)
            outer_c, outer_r = sorted_rois[0]
            inner_c, inner_r = sorted_rois[1]

            cv2.circle(mask, outer_c, outer_r, 255, -1)
            cv2.circle(mask, inner_c, inner_r, 0, -1)

            ref_center = outer_c
            ref_radius = outer_r
        else:
            return frame

        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return np.full(
                (ref_radius * 2 + 1, ref_radius * 2 + 1, 3), 255, dtype=np.uint8
            )

        shape = (
            (ref_radius * 2 + 1, ref_radius * 2 + 1)
            if frame.ndim == 2
            else (ref_radius * 2 + 1, ref_radius * 2 + 1, frame.shape[2])
        )
        new_frame = np.full(shape, 255, dtype=np.uint8)

        y_indices = ys - ref_center[1] + ref_radius
        x_indices = xs - ref_center[0] + ref_radius

        valid = (
            (y_indices >= 0)
            & (y_indices < shape[0])
            & (x_indices >= 0)
            & (x_indices < shape[1])
        )

        new_frame[y_indices[valid], x_indices[valid]] = frame[ys[valid], xs[valid]]
        return new_frame

    def modal(self):
        """
        Display the current frame in grayscale mode.
        Used during preprocessing and other modal operations.
        """
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(self.window, frame)
        cv2.waitKey(1)

    def preprocess(self, rois):
        """
        Preprocesses the video to identify potential rearing events.
        Applies ROI extraction (circle or annulus), downsampling, color thresholding, and peak detection.
        Saves detected event properties for navigation and review.

        Args:
            rois (list): List of (center, radius) tuples defining the ROI
        """
        self.modal()
        downsample_rate = int(self.fps / 5)
        dim = (128, 128)
        stats = pd.DataFrame(
            np.zeros(int(np.ceil(self.frame_count / downsample_rate))),
            columns=["percentage"],
        )

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            ret = False
            while not ret:
                ret, frame = self.video.read()
            if i % downsample_rate != 0:
                continue
            frame = self.extract_roi(frame, rois)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_NEAREST)
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            stats.loc[i / downsample_rate, "percentage"] = (
                np.sum(mask) / np.prod(dim) / 255
            )

        signal = (
            -stats["percentage"] if self.view_mode == "bottom" else stats["percentage"]
        )
        peaks, properties = find_peaks(
            signal,
            prominence=max(stats["percentage"]) * 0.3,
            width=(None, self.fps * 5),
            rel_height=0.5,
        )
        self.properties = pd.DataFrame(properties) * downsample_rate
        self.properties.to_csv(self.properties_path, index=False)
        self.update_frame_index(0)
        return

    def create_window(self):
        """
        Creates and configures the main OpenCV window for video display and interaction.
        Sets up window size, frame navigation trackbar, and mouse event callbacks.
        """
        cv2.namedWindow(self.window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(
            self.window,
            int(self.width * self.resize_factor),
            int(self.height * self.resize_factor),
        )
        cv2.createTrackbar(
            "Frame",
            self.window,
            self.frame_index,
            self.frame_count - 1,
            self.update_frame_index,
        )
        cv2.setMouseCallback(self.window, self.click_event)
        return

    def update_frame_index(self, frame_index):
        """
        Updates the current frame index, wrapping around if necessary.
        Used for all navigation actions.

        Args:
            frame_index (int): New frame index to set
        """
        self.frame_index = int(frame_index % self.frame_count)
        return

    def update_frame(self):
        """
        Loads the current frame from the video and updates the display.
        Ensures the frame is valid before proceeding.
        """
        cv2.setTrackbarPos("Frame", self.window, self.frame_index)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret = False
        while not ret:
            ret, self.frame = self.video.read()
        return

    def print_frame(self):
        """
        Displays the current frame in the OpenCV window.
        """
        cv2.imshow(self.window, self.frame)
        return

    def monitor_key(self):
        """
        Handles all keyboard input for navigation, labeling, saving, and tool control.

        Key bindings:
            - Space: Toggle play/fast-forward mode
            - w/a/s/d: Label as none/left/both/right rearing
            - 1-9: Set step size for navigation/labeling
            - q/e: Step backward/forward by current step size
            - r: Re-encode current video
            - f: Open filter editor
            - t: Toggle view mode (Bottom/Top)
            - x: Save current score
            - v: Take screenshot of current frame
            - ESC/Enter: Exit tool (score is always saved)
            - z/c: Jump to previous/next potential rearing event
        """
        if self.key == " ":
            self.play = not self.play

        elif self.key == "w":
            for i in range(self.step):
                self.score.loc[self.frame_index, "label"] = "n"
                self.update_frame_index(self.frame_index + 1)
        elif self.key == "a":
            for i in range(self.step):
                self.score.loc[self.frame_index, "label"] = "l"
                self.update_frame_index(self.frame_index + 1)
        elif self.key == "s":
            for i in range(self.step):
                self.score.loc[self.frame_index, "label"] = "b"
                self.update_frame_index(self.frame_index + 1)
        elif self.key == "d":
            for i in range(self.step):
                self.score.loc[self.frame_index, "label"] = "r"
                self.update_frame_index(self.frame_index + 1)

        elif self.key == "r":
            self.reencoder()
        elif self.key == "f":
            self.filter_editor()
        elif self.key == "t":
            self.view_mode = "top" if self.view_mode == "bottom" else "bottom"
            self.rois = []

        elif self.key and self.key.isdigit() and 0 < int(self.key) < 10:
            self.step = int(self.key)
        elif self.key == "q":
            self.update_frame_index(self.frame_index - self.step)
        elif self.key == "e":
            self.update_frame_index(self.frame_index + self.step)
        elif self.key == "x":
            self.save()
        elif self.key == "v":
            self.screenshot()
        elif ord(self.key) == 27:
            self.run = False
        elif ord(self.key) == 13:
            self.run = False
        elif (
            self.key == "c"
            and self.properties is not None
            and not self.properties.empty
        ):
            frame_index = (
                self.properties[
                    self.properties["left_ips"].gt(self.frame_index + 1)
                ].first_valid_index()
                or 0
            )
            self.update_frame_index(self.properties.loc[frame_index, "left_ips"])
        elif (
            self.key == "z"
            and self.properties is not None
            and not self.properties.empty
        ):
            frame_index = self.properties[
                self.properties["left_ips"].lt(self.frame_index)
            ].last_valid_index() or -1 % len(self.properties)
            self.update_frame_index(self.properties.loc[frame_index, "left_ips"])

        if ord(self.key) != 255:
            self.stamp = [time(), self.frame_index]
        return

    def resize(self, num):
        """
        Utility function to scale UI elements according to the resize factor.
        Ensures consistent appearance regardless of video resolution.

        Args:
            num (int): Number to scale

        Returns:
            int: Scaled number
        """
        return int(num / self.resize_factor)

    def draw_circle(self):
        """
        Handles interactive selection and drawing of the ROI for preprocessing.
        Draws green circle overlays during selection.
        In 'bottom' mode: triggers preprocessing after one circle.
        In 'top' mode: triggers preprocessing after two circles (annulus).
        """
        for center, radius in self.rois:
            cv2.circle(self.frame, center, radius, (0, 255, 0), 1)

        if self.mouse_start:
            end = self.mouse_end or self.mouse
            current_center = (
                int((self.mouse_start[0] + end[0]) / 2),
                int((self.mouse_start[1] + end[1]) / 2),
            )
            current_radius = int(
                np.sqrt(
                    (end[0] - self.mouse_start[0]) ** 2
                    + (end[1] - self.mouse_start[1]) ** 2
                )
                / 2,
            )
            cv2.circle(self.frame, current_center, current_radius, (0, 255, 0), 1)

            if self.mouse_end:
                self.rois.append((current_center, current_radius))
                self.mouse_start = None
                self.mouse_end = None

                if self.view_mode == "bottom":
                    self.preprocess(self.rois)
                    self.rois = []
                elif self.view_mode == "top" and len(self.rois) == 2:
                    self.preprocess(self.rois)
                    self.rois = []
        return

    def draw_overlay(self):
        """
        Draws all user interface overlays on the current frame, including:
            - Step size indicator (bottom left)
            - Current, previous, and next frame labels (center, left, right)
        Label colors and layout are consistent with the scoring scheme.
        """
        cv2.rectangle(
            self.frame,
            (self.resize(10), self.height - self.resize(70)),
            (self.resize(70), self.height - self.resize(10)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            self.frame,
            str(self.step),
            (self.resize(20), self.height - self.resize(20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.resize(2),
            (255, 255, 255),
            self.resize(4),
        )

        cv2.rectangle(
            self.frame,
            (self.width - self.resize(70), self.height - self.resize(70)),
            (self.width - self.resize(10), self.height - self.resize(10)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            self.frame,
            self.view_mode[0].upper(),
            (self.width - self.resize(60), self.height - self.resize(20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.resize(2),
            (255, 255, 255),
            self.resize(4),
        )

        cv2.rectangle(
            self.frame,
            (int(self.width / 2) - self.resize(30), self.resize(30)),
            (int(self.width / 2) + self.resize(30), self.resize(90)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            self.frame,
            self.score.loc[self.frame_index, "label"].upper(),
            (int(self.width / 2) - self.resize(20), self.resize(80)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.resize(2),
            self.colors[self.score.loc[self.frame_index, "label"]],
            self.resize(4),
        )

        cv2.rectangle(
            self.frame,
            (self.resize(10), self.resize(30)),
            (self.resize(70), self.resize(90)),
            (0, 0, 0),
            -1,
        )
        if self.frame_index:
            cv2.putText(
                self.frame,
                self.score.loc[self.frame_index - 1, "label"].upper(),
                (self.resize(20), self.resize(80)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.resize(2),
                self.colors[self.score.loc[self.frame_index - 1, "label"]],
                self.resize(4),
            )

        cv2.rectangle(
            self.frame,
            (self.width - self.resize(70), self.resize(30)),
            (self.width - self.resize(10), self.resize(90)),
            (0, 0, 0),
            -1,
        )
        if self.frame_count - self.frame_index - 1:
            cv2.putText(
                self.frame,
                self.score.loc[self.frame_index + 1, "label"].upper(),
                (self.width - self.resize(60), self.resize(80)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.resize(2),
                self.colors[self.score.loc[self.frame_index + 1, "label"]],
                self.resize(4),
            )
        return

    def screenshot(self):
        """
        Saves a screenshot of the current frame for documentation or review.
        The screenshot filename includes the current frame index.
        """
        screenshot_path = self.screenshot_path.with_name(
            f"{self.screenshot_path.stem}_{self.frame_index}{self.screenshot_path.suffix}"
        )
        cv2.imwrite(str(screenshot_path), self.frame)
        return

    def draw_peaks(self):
        """
        Draws the timeline bar at the top of the frame, visualizing scoring and event status:
            - Green triangle: current frame position
            - Colored bars: labeled (scored) rearing events
            - Gray bars: potential rearing events (to be scored or unscored)
        """
        height = self.resize(20)
        cv2.rectangle(
            self.frame,
            (0, 0),
            (self.width - 1, height),
            (0, 0, 0),
            -1,
        )
        if self.properties is not None:
            for i in range(len(self.properties)):
                cv2.rectangle(
                    self.frame,
                    (
                        int(
                            self.properties.loc[i, "left_ips"]
                            / self.frame_count
                            * self.width
                        ),
                        0,
                    ),
                    (
                        int(
                            self.properties.loc[i, "right_ips"]
                            / self.frame_count
                            * self.width
                        ),
                        height,
                    ),
                    (63, 63, 63),
                    -1,
                )
        frame_indexes = self.score[self.score["label"].ne("n")].index
        for i in frame_indexes:
            cv2.line(
                self.frame,
                (
                    int(i / self.frame_count * self.width),
                    0,
                ),
                (
                    int(i / self.frame_count * self.width),
                    height,
                ),
                self.colors[self.score.loc[i, "label"]],
            )
        cv2.fillPoly(
            self.frame,
            np.array(
                [
                    [
                        [
                            int(self.frame_index / self.frame_count * self.width),
                            height + 1,
                        ],
                        [
                            int(self.frame_index / self.frame_count * self.width)
                            - self.resize(5),
                            height + self.resize(6),
                        ],
                        [
                            int(self.frame_index / self.frame_count * self.width)
                            + self.resize(5),
                            height + self.resize(6),
                        ],
                    ]
                ]
            ),
            (0, 255, 0),
        )
        return

    def click_event(self, event, x, y, flags, param):
        """
        Handles all mouse events for frame navigation and ROI selection.
        - Shift+Left click drag: select circular ROI for preprocessing
        - Left button release: complete ROI selection
        - Right click: cancel current drag or clear confirmed ROIs

        Args:
            event: OpenCV mouse event type
            x (int): Mouse x coordinate
            y (int): Mouse y coordinate
            flags: OpenCV event flags
            param: Additional parameters (unused)
        """
        self.mouse = (x, y)
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    self.mouse_start = (x, y)
            case cv2.EVENT_RBUTTONDOWN:
                if self.mouse_start:
                    self.mouse_start = None
                else:
                    self.rois = []
            case cv2.EVENT_LBUTTONUP:
                if self.mouse_start:
                    self.mouse_end = (x, y)
        return

    def reencoder(self):
        """
        Re-encodes the current video with specified parameters.
        Closes current session and opens the newly encoded video.
        """
        self.modal()

        total_seconds = int(self.frame_count / self.fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        video_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        duration = simpledialog.askstring(
            "Re-encode Duration",
            f"Enter duration (HH:MM:SS).\nVideo length: {video_duration}",
            initialvalue=video_duration,
        )
        if not duration:
            return

        new_video_path = self.video_path.with_name(
            f"{self.video_path.stem}_reencoded.mp4"
        )
        self.reencode_video(self.video_path, new_video_path, duration)
        self.close()
        ScoreCR(new_video_path)

    def filter_editor(self):
        """
        Opens a filter editor window to adjust HSV color thresholds.
        Provides real-time preview of filter effects with trackbars for HSV parameters.
        ESC to exit, Enter to apply changes.
        """
        self.modal()
        filter_window = "Filter Editor"
        cv2.namedWindow(filter_window, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(
            filter_window,
            int(self.width * self.resize_factor),
            int(self.height * self.resize_factor / 2),
        )

        cv2.createTrackbar(
            "Hue Min", filter_window, self.hsv_lower[0], 179, lambda x: None
        )
        cv2.createTrackbar(
            "Hue Max", filter_window, self.hsv_upper[0], 179, lambda x: None
        )
        cv2.createTrackbar(
            "Sat Min", filter_window, self.hsv_lower[1], 255, lambda x: None
        )
        cv2.createTrackbar(
            "Sat Max", filter_window, self.hsv_upper[1], 255, lambda x: None
        )
        cv2.createTrackbar(
            "Val Min", filter_window, self.hsv_lower[2], 255, lambda x: None
        )
        cv2.createTrackbar(
            "Val Max", filter_window, self.hsv_upper[2], 255, lambda x: None
        )

        while True:
            try:
                if cv2.getWindowProperty(filter_window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            try:
                h_min = cv2.getTrackbarPos("Hue Min", filter_window)
                h_max = cv2.getTrackbarPos("Hue Max", filter_window)
                s_min = cv2.getTrackbarPos("Sat Min", filter_window)
                s_max = cv2.getTrackbarPos("Sat Max", filter_window)
                v_min = cv2.getTrackbarPos("Val Min", filter_window)
                v_max = cv2.getTrackbarPos("Val Max", filter_window)
            except cv2.error:
                break

            self.hsv_lower = (h_min, s_min, v_min)
            self.hsv_upper = (h_max, s_max, v_max)

            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            result = cv2.bitwise_and(self.frame, self.frame, mask=mask)

            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([self.frame, mask_colored, result])

            try:
                cv2.imshow(filter_window, combined)
            except cv2.error:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 13:
                break

        cv2.destroyWindow(filter_window)
        return

    def reencode_video(self, video_path, new_video_path, duration="00:05:00"):
        """
        Re-encodes a video file with specified parameters.
        Crops to square aspect ratio, scales to 720x720, and limits to specified duration at 15fps.

        Args:
            video_path (str): Path to input video file
            new_video_path (str): Path to output video file
            duration (str): Duration of the output video (HH:MM:SS)
        """
        probe = ffmpeg.probe(video_path)
        video_info = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ][0]
        width = video_info["width"]
        height = video_info["height"]
        crop_size = min(width, height)
        if width > height:
            x_offset = (width - crop_size) // 2
            y_offset = 0
        else:
            x_offset = 0
            y_offset = (height - crop_size) // 2
        (
            ffmpeg.input(video_path)
            .filter("crop", crop_size, crop_size, x_offset, y_offset)
            .filter("scale", 720, 720)
            .output(str(new_video_path), vcodec="libx264", t=duration, r=15)
            .overwrite_output()
            .run()
        )
        return

    def save(self):
        """
        Saves the current scoring results to the score CSV file.
        The file includes frame-wise labels and summary counts for each label type.
        """
        all_labels = [col for col in self.score.columns if col != "label"]
        for label in all_labels:
            self.score.loc[0, label] = 0
            self.score.loc[1, label] = 0

        for i in set(self.score["label"]):
            self.score.loc[0, i] = self.score["label"].value_counts()[i]

        labels = self.score["label"].tolist()
        runs = {}
        if labels:
            current_label = labels[0]
            for label in labels[1:]:
                if label != current_label:
                    runs[current_label] = runs.get(current_label, 0) + 1
                    current_label = label
            runs[current_label] = runs.get(current_label, 0) + 1

        for label, count in runs.items():
            self.score.loc[1, label] = count

        self.score.to_csv(self.score_path, index=False)
        return

    def close(self):
        """
        Finalizes the session: saves results, releases video resources, and closes all windows.
        """
        self.save()
        self.video.release()
        cv2.destroyAllWindows()
        return


if __name__ == "__main__":
    ScoreCR()
