import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "/Users/lucasmarch/Projects/Clarinet_Fingerings/hand_landmarker.task"
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


# @markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import csv


pose_hand = [
    "WRIST",
    "THUMB_CPC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "RING_FINGER_MCP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

pose_hand_2 = [
    "WRIST2",
    "THUMB_CPC2",
    "THUMB_MCP2",
    "THUMB_IP2",
    "THUMB_TIP2",
    "INDEX_FINGER_MCP2",
    "INDEX_FINGER_PIP2",
    "INDEX_FINGER_DIP2",
    "INDEX_FINGER_TIP2",
    "MIDDLE_FINGER_MCP2",
    "MIDDLE_FINGER_PIP2",
    "MIDDLE_FINGER_DIP2",
    "MIDDLE_FINGER_TIP2",
    "RING_FINGER_PIP2",
    "RING_FINGER_DIP2",
    "RING_FINGER_TIP2",
    "RING_FINGER_MCP2",
    "PINKY_MCP2",
    "PINKY_PIP2",
    "PINKY_DIP2",
    "PINKY_TIP2",
]


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None
        self.all_data = {"Right": [], "Left": []}

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        # hand_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        # Loop through the detected handss to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hands landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
            )
        return annotated_image

    # Create a hands landmarker instance with the live stream mode:

    def print_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ):
        print("hands landmarker result: {}".format(result))
        self.results = result

        frame_data = {"Right": {}, "Left": {}}

        # Iterate over each hand landmarks list
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = "Right" if i == 0 else "Left"
            for j, landmark in enumerate(hand_landmarks):
                # Accessing landmarks for each hand
                landmark_name = pose_hand[j]
                x = landmark.x * output_image.width
                y = landmark.y * output_image.height
                z = landmark.z
                frame_data[handedness][landmark_name] = {"x": x, "y": y, "z": z}

        # Append frame data to the list
        self.all_data["Right"].append(frame_data["Right"])
        self.all_data["Left"].append(frame_data["Left"])

    def save_to_csv(self, file_path):
        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["Frame", "Hand", "Landmark", "x", "y", "z"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for frame_num, (right_data, left_data) in enumerate(
                zip(self.all_data["Right"], self.all_data["Left"])
            ):
                # Check if frame_data is empty (no landmarks detected)
                if not right_data and not left_data:
                    continue  # Skip this frame

                for handedness, data in [("Right", right_data), ("Left", left_data)]:
                    for landmark, coordinates in data.items():
                        writer.writerow(
                            {
                                "Frame": frame_num,
                                "Hand": handedness,
                                "Landmark": landmark,
                                "x": coordinates["x"],
                                "y": coordinates["y"],
                                "z": coordinates["z"],
                            }
                        )

    def main(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.print_result,
        )

        video = cv2.VideoCapture(0)

        timestamp = 0
        with HandLandmarker.create_from_options(options) as landmarker:
            # The landmarker is initialized. Use it here.
            # ...
            while video.isOpened():
                # Capture frame-by-frame
                ret, frame = video.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, timestamp)

                if not (self.results is None):
                    annotated_image = self.draw_landmarks_on_image(
                        mp_image.numpy_view(), self.results
                    )
                    # cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Show", annotated_image)
                else:
                    cv2.imshow("Show", frame)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    print("Closing Camera Stream")
                    break

            video.release()
            cv2.destroyAllWindows()

            # Save data to CSV after processing all frames
            self.save_to_csv("hand_landmarks_data.csv")


if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()
