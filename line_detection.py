import cv2
import numpy as np


# -----------------------------
# Parameters
# -----------------------------

# Canny edge detection
CANNY_LOW = 50
CANNY_HIGH = 150

# Probabilistic Hough transform
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

# Width of each display panel
DISPLAY_WIDTH = 480


# -----------------------------
# Canny Edge Detection
# -----------------------------

def detect_canny(frame):
    """
    Apply Canny edge detection to a frame.

    Returns:
        BGR image containing the Canny edge map.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    # Convert grayscale image back to BGR so it can
    # be displayed alongside color images.
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# -----------------------------
# Hough Line Detection
# -----------------------------

def detect_hough(frame):
    """
    Detect line segments using the probabilistic Hough transform.

    Returns:
        Copy of the original frame with detected line segments drawn.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    result = frame.copy()

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is not None:
        # OpenCV versions can return Hough lines in slightly
        # different array shapes, so reshape everything into
        # rows of [x1, y1, x2, y2].
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            cv2.line(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

    return result


# -----------------------------
# Display Helpers
# -----------------------------

def add_label(image, label):
    """
    Add a label to the top-left corner of an image.
    """
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


def resize_frame(frame, width=DISPLAY_WIDTH):
    """
    Resize an image while preserving its aspect ratio.
    """
    height = int(
        frame.shape[0] * (width / frame.shape[1])
    )

    return cv2.resize(
        frame,
        (width, height)
    )


# -----------------------------
# Main Program
# -----------------------------

def main():
    """
    Open the webcam and run the real-time
    edge and line detection system.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")
    print("Press Q to quit.")

    while True:

        # -------------------------
        # Capture frame
        # -------------------------

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # -------------------------
        # Run detection methods
        # -------------------------

        canny = detect_canny(frame)
        hough = detect_hough(frame)

        # -------------------------
        # Add panel labels
        # -------------------------

        original_display = add_label(
            frame,
            "Original"
        )

        canny_display = add_label(
            canny,
            "Canny"
        )

        hough_display = add_label(
            hough,
            "Hough"
        )

        # -------------------------
        # Resize panels
        # -------------------------

        original_display = resize_frame(
            original_display
        )

        canny_display = resize_frame(
            canny_display
        )

        hough_display = resize_frame(
            hough_display
        )

        # -------------------------
        # Combine panels
        # -------------------------

        display = np.hstack(
            (
                original_display,
                canny_display,
                hough_display
            )
        )

        # -------------------------
        # Display
        # -------------------------

        cv2.imshow(
            "Real-Time Line Detection",
            display
        )

        # -------------------------
        # Quit
        # -------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # -----------------------------
    # Clean up
    # -----------------------------

    cap.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


# -----------------------------
# Program Entry Point
# -----------------------------

if __name__ == "__main__":
    main()