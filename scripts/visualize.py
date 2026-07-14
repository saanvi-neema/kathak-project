"""
Phase 1: Landmark Visualization
Overlays pose and hand landmarks on video for visual validation.
Run this on one clip from each category before processing everything.

Usage:
    python visualize.py --video data/clips/teacher_tatkaar_01.mp4
    python visualize.py --video data/clips/teacher_tatkaar_01.mp4 --save   # save output video
"""

import argparse
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


def visualize(video_path: str, save: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        output_path = video_path.replace("clips/", "landmarks/").replace(".mp4", "_overlay.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Saving overlay video to {output_path}")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            hand_result = hands.process(rgb)

            # Draw body landmarks
            if pose_result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style(),
                )

            # Draw hand landmarks
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_styles.get_default_hand_landmarks_style(),
                        mp_draw_styles.get_default_hand_connections_style(),
                    )

            if writer:
                writer.write(frame)

            cv2.imshow("Landmark Overlay — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize pose and hand landmarks on video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--save", action="store_true", help="Save overlay video to data/landmarks/")
    args = parser.parse_args()

    visualize(args.video, save=args.save)


if __name__ == "__main__":
    main()
