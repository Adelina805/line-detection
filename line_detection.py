import cv2
import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

# Least Squares
LS_REGION_TOP = 0.30
LS_REGION_COUNT = 4
LS_MIN_POINTS = 40
LS_MAX_LINES = 3

# Display
DISPLAY_WIDTH = 480


# ============================================================
# CANNY EDGE DETECTION
# ============================================================

def get_edges(frame):
    """
    Convert a frame to grayscale and compute its Canny edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )


def detect_canny(frame):
    """
    Return the Canny edge map as a BGR image.
    """
    edges = get_edges(frame)

    return cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )


# ============================================================
# HOUGH LINE DETECTION
# ============================================================

def detect_hough(frame):
    """
    Detect line segments using the probabilistic Hough transform.
    """

    edges = get_edges(frame)

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


# ============================================================
# LEAST-SQUARES LINE DETECTION
# ============================================================

def fit_line(points):
    """
    Fit x = a*y + b to a collection of points.

    Using x as the dependent variable allows the fitted
    representation to handle vertical lines.
    """

    x = points[:, 0]
    y = points[:, 1]

    A = np.column_stack(
        (y, np.ones(len(y)))
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        A,
        x,
        rcond=None
    )

    a, b = coefficients

    return a, b


def detect_least_squares(frame):
    """
    Detect line structures by fitting least-squares lines
    to edge pixels within multiple image regions.
    """

    edges = get_edges(frame)

    result = frame.copy()

    height, width = edges.shape

    # Only use the lower portion of the frame.
    region_top = int(height * LS_REGION_TOP)

    region_height = height - region_top
    region_width = width / LS_REGION_COUNT

    candidates = []

    # --------------------------------------------------------
    # Divide the fitting region into vertical sections.
    # --------------------------------------------------------

    for i in range(LS_REGION_COUNT):

        x_start = int(i * region_width)
        x_end = int((i + 1) * region_width)

        region = edges[
            region_top:height,
            x_start:x_end
        ]

        # Find edge pixels
        y_coords, x_coords = np.nonzero(region)

        if len(x_coords) < LS_MIN_POINTS:
            continue

        # Convert coordinates back to full-frame coordinates.
        x_coords = x_coords + x_start
        y_coords = y_coords + region_top

        points = np.column_stack(
            (x_coords, y_coords)
        )

        # Fit a line
        a, b = fit_line(points)

        candidates.append(
            {
                "a": a,
                "b": b,
                "points": len(points)
            }
        )

    # --------------------------------------------------------
    # Keep only the strongest candidate regions.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda candidate: candidate["points"],
        reverse=True
    )

    candidates = candidates[:LS_MAX_LINES]

    # --------------------------------------------------------
    # Draw fitted lines.
    # --------------------------------------------------------

    for candidate in candidates:

        a = candidate["a"]
        b = candidate["b"]

        y1 = region_top
        y2 = height - 1

        x1 = int(a * y1 + b)
        x2 = int(a * y2 + b)

        # Skip lines that are far outside the image.
        if (
            x1 < -width or
            x1 > 2 * width or
            x2 < -width or
            x2 > 2 * width
        ):
            continue

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

    # --------------------------------------------------------
    # Draw the top boundary of the fitting region.
    # --------------------------------------------------------

    cv2.line(
        result,
        (0, region_top),
        (width, region_top),
        (255, 255, 0),
        1
    )

    return result


# ============================================================
# DISPLAY HELPERS
# ============================================================

def add_label(image, label):
    """
    Add a label to the top-left corner.
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
    Resize while preserving aspect ratio.
    """

    height = int(
        frame.shape[0] * (width / frame.shape[1])
    )

    return cv2.resize(
        frame,
        (width, height)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")
    print("Press Q to quit.")

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        canny = detect_canny(frame)

        hough = detect_hough(frame)

        least_squares = detect_least_squares(frame)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

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

        least_squares_display = add_label(
            least_squares,
            "Least Squares"
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        original_display = resize_frame(
            original_display
        )

        canny_display = resize_frame(
            canny_display
        )

        hough_display = resize_frame(
            hough_display
        )

        least_squares_display = resize_frame(
            least_squares_display
        )

        # ----------------------------------------------------
        # 2 x 2 layout
        # ----------------------------------------------------

        top_row = np.hstack(
            (
                original_display,
                canny_display
            )
        )

        bottom_row = np.hstack(
            (
                hough_display,
                least_squares_display
            )
        )

        display = np.vstack(
            (
                top_row,
                bottom_row
            )
        )

        # ----------------------------------------------------
        # Show result
        # ----------------------------------------------------

        cv2.imshow(
            "Real-Time Line Detection",
            display
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


if __name__ == "__main__":
    main()