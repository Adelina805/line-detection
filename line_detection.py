import cv2
import numpy as np
import time


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
DISPLAY_WIDTH = 800
FONT_SCALE = 1.2
FONT_THICKNESS = 3
TEXT_MARGIN = 15
TEXT_PADDING = 10
WINDOW_NAME = "Real-Time Line Detection"


# ============================================================
# CANNY EDGE DETECTION
# ============================================================

def get_edges(frame):
    """
    Convert a frame to grayscale and compute Canny edges.
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

    # Scale thickness so lines stay visible after downsizing.
    thickness = max(2, frame.shape[1] // 300)

    if lines is not None:

        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            cv2.line(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                thickness
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

    region_width = width / LS_REGION_COUNT

    candidates = []

    # --------------------------------------------------------
    # Divide the fitting area into vertical regions
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

        # Fit a least-squares line.
        a, b = fit_line(points)

        candidates.append(
            {
                "a": a,
                "b": b,
                "points": len(points)
            }
        )

    # --------------------------------------------------------
    # Keep strongest candidate regions
    # --------------------------------------------------------

    candidates.sort(
        key=lambda candidate: candidate["points"],
        reverse=True
    )

    candidates = candidates[:LS_MAX_LINES]

    # --------------------------------------------------------
    # Draw fitted lines
    # --------------------------------------------------------

    for candidate in candidates:

        a = candidate["a"]
        b = candidate["b"]

        y1 = region_top
        y2 = height - 1

        x1 = int(a * y1 + b)
        x2 = int(a * y2 + b)

        # Skip lines far outside the image.
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
            max(3, width // 250)
        )

    # Draw the boundary of the fitting region.
    cv2.line(
        result,
        (0, region_top),
        (width, region_top),
        (255, 255, 0),
        max(1, width // 600)
    )

    return result


# ============================================================
# DISPLAY HELPERS
# ============================================================

def draw_text_box(image, text, color, align_right=False):
    """
    Draw text on a filled dark box in a top corner of the image.
    """

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_THICKNESS
    )

    if align_right:
        x = image.shape[1] - text_width - TEXT_MARGIN - TEXT_PADDING
    else:
        x = TEXT_MARGIN + TEXT_PADDING

    y = TEXT_MARGIN + TEXT_PADDING + text_height

    cv2.rectangle(
        image,
        (x - TEXT_PADDING, y - text_height - TEXT_PADDING),
        (x + text_width + TEXT_PADDING, y + baseline + TEXT_PADDING),
        (0, 0, 0),
        cv2.FILLED
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv2.LINE_AA
    )


def add_label(image, label):
    """
    Add a label to the top-left corner.
    """

    result = image.copy()

    draw_text_box(result, label, (0, 255, 0))

    return result


def add_fps(image, fps):
    """
    Display the current smoothed FPS in the upper-right corner.
    """

    result = image.copy()

    draw_text_box(
        result,
        f"FPS: {fps:.1f}",
        (0, 255, 255),
        align_right=True
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

    # Resizable window, initially sized to fit the 2 x 2 grid.
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 16
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 9
    panel_height = int(frame_height * (DISPLAY_WIDTH / frame_width))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 2 * DISPLAY_WIDTH, 2 * panel_height)

    # Used to calculate FPS.
    previous_time = time.perf_counter()

    # Start at zero and gradually smooth the measurement.
    smoothed_fps = 0.0

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # ----------------------------------------------------
        # Run detection methods
        # ----------------------------------------------------

        canny = detect_canny(frame)

        hough = detect_hough(frame)

        least_squares = detect_least_squares(frame)

        # ----------------------------------------------------
        # Calculate FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        elapsed_time = current_time - previous_time

        previous_time = current_time

        if elapsed_time > 0:

            current_fps = 1.0 / elapsed_time

            # Exponential smoothing prevents the displayed
            # FPS value from jumping dramatically every frame.
            if smoothed_fps == 0:
                smoothed_fps = current_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * current_fps
                )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        original_display = resize_frame(frame)

        canny_display = resize_frame(canny)

        hough_display = resize_frame(hough)

        least_squares_display = resize_frame(least_squares)

        # ----------------------------------------------------
        # Add labels (after resizing so text stays full size)
        # ----------------------------------------------------

        original_display = add_label(
            original_display,
            "Original"
        )

        # Show FPS on the original panel.
        original_display = add_fps(
            original_display,
            smoothed_fps
        )

        canny_display = add_label(
            canny_display,
            "Canny"
        )

        hough_display = add_label(
            hough_display,
            "Hough"
        )

        least_squares_display = add_label(
            least_squares_display,
            "Least Squares"
        )

        # ----------------------------------------------------
        # Create 2 x 2 layout
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
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
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