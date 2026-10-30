"""Image processing for the live preview."""

import cv2


def highlight_edges(frame, sensitivity):
    """Draw contrast edges in BGR purple on a camera frame, in place."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Higher sensitivity lowers the Canny thresholds to reveal weaker edges.
    high = 260 - 2 * max(1, min(100, sensitivity))
    edges = cv2.Canny(blurred, high // 2, high)
    edges = cv2.dilate(edges, None, iterations=1)
    frame[edges != 0] = (180, 0, 140)  # BGR
    return frame


