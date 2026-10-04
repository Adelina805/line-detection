import cv2
import numpy as np


CANNY_LOW = 50
CANNY_HIGH = 150


def detect_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # Convert back to BGR so it can be displayed beside color images later
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def add_label(image, label):
    result = image.copy()

    cv2.putText(
        result,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return result


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        canny = detect_canny(frame)

        original_display = add_label(frame, "Original")
        canny_display = add_label(canny, "Canny")

        display = np.hstack((original_display, canny_display))

        cv2.imshow("Real-Time Line Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()