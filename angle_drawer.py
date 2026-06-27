from pathlib import Path

import cv2

from body_detector import body_detector


def process_video(input_path: str, output_path: str) -> None:
    detector = body_detector()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Change the fourcc codec to 'h264' (or 'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'h264') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results, _ = detector.detect_body(frame)
            frame = detector.draw_body_landmarks(frame, results)
            angle = detector.calc_angle_esh(frame, results)

            angle_text = f"Angle: {angle:.1f} deg" if angle is not None else "Angle: --"
            cv2.putText(
                frame,
                angle_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    print(f"Processed {frame_count} frames")
    print(f"Saved output to: {output_path}")


def main() -> None:
    input_path = Path("exercises/shoulder_abduction.mp4")  # change this path

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = str(input_path.with_name(f"{input_path.stem}_with_angles.mp4"))

    process_video(str(input_path), output_path)


if __name__ == "__main__":
    main()