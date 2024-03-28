import mediapipe as mp
import cv2
from mediapipe import solutions
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import csv
from joblib import load
from sklearn.preprocessing import StandardScaler

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


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


class HandFingerClassifier:
    def __init__(self):
        self.knn_classifier = load("models/knn_classifier.joblib")
        self.svm_classifier = load("models/svm_classifier.joblib")
        self.scaler = load("models/std_scaler.joblib")

    def predict_fingering(self, hand_landmarks, output_image):
        features = self.extract_features(hand_landmarks, output_image)
        if features is not None:
            normalized_features = self.scaler.transform(features)
            knn_prediction = self.knn_classifier.predict(normalized_features)
            svm_prediction = self.svm_classifier.predict(normalized_features)
            return knn_prediction, svm_prediction

        return None, None

    def extract_features(self, result, output_image):

        if len(result.hand_landmarks) != 2 or any(
            len(hand) != 21 for hand in result.hand_landmarks
        ):
            return None  # Skip this result
        # Dictionary to store coordinates of each landmark for both left and right hand
        hand_coordinates = {"Right": {}, "Left": {}}

        # Iterate over each hand landmarks list
        for i, landmarks_list in enumerate(result.hand_landmarks):
            handedness = "Right" if i == 0 else "Left"
            for j, landmark in enumerate(landmarks_list):
                # Calculate coordinates
                x = landmark.x * output_image.width
                y = landmark.y * output_image.height
                z = landmark.z
                # Store coordinates in dictionary
                hand_coordinates[handedness][j] = (x, y, z)

        # Construct feature names and append coordinates
        features = []
        for handedness in ["Left", "Right"]:
            for j in range(21):
                x, y, z = hand_coordinates[handedness][j]
                features.extend([x, y, z])
        print(np.array([features]).shape)
        return np.array([features])


class Mediapipe_BodyModule:
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.results = None
        self.hand_classifier = HandFingerClassifier()

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
        self.results = result
        # Predict fingering for each hand
        knn_prediction, svm_prediction = self.hand_classifier.predict_fingering(
            self.results, output_image
        )
        print("KNN Prediction:", knn_prediction)
        print("SVM Prediction:", svm_prediction)

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


if __name__ == "__main__":
    body_module = Mediapipe_BodyModule()
    body_module.main()
