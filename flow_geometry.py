"""Decision line and flow direction, for a line of either orientation.

A vertical line (x = position) is crossed by cashews moving right-to-left or left-to-right;
a horizontal line (y = position) by cashews moving top-to-bottom or bottom-to-top.
Everything else in the grader works with `progress`: how many pixels a point is past the line
in the direction of flow (negative = still before the line).
"""
import cv2
import numpy as np

RIGHT_TO_LEFT, LEFT_TO_RIGHT = "right_to_left", "left_to_right"
TOP_TO_BOTTOM, BOTTOM_TO_TOP = "top_to_bottom", "bottom_to_top"
FLOWS = (RIGHT_TO_LEFT, LEFT_TO_RIGHT, TOP_TO_BOTTOM, BOTTOM_TO_TOP)
FLOW_LABELS = {RIGHT_TO_LEFT: "Right → Left", LEFT_TO_RIGHT: "Left → Right",
               TOP_TO_BOTTOM: "Top → Bottom", BOTTOM_TO_TOP: "Bottom → Top"}
VERTICAL, HORIZONTAL = "vertical", "horizontal"         # orientation of the LINE
FLOWS_FOR_LINE = {VERTICAL: (RIGHT_TO_LEFT, LEFT_TO_RIGHT), HORIZONTAL: (TOP_TO_BOTTOM, BOTTOM_TO_TOP)}


def line_orientation(flow) -> str:
    return VERTICAL if flow in FLOWS_FOR_LINE[VERTICAL] else HORIZONTAL


def valid_flow(flow) -> str:
    return flow if flow in FLOWS else RIGHT_TO_LEFT


def _unit(flow):
    """Direction of travel as (dx, dy)."""
    return {RIGHT_TO_LEFT: (-1, 0), LEFT_TO_RIGHT: (1, 0), TOP_TO_BOTTOM: (0, 1), BOTTOM_TO_TOP: (0, -1)}[flow]


def progress(flow, line_pos, cx, cy) -> float:
    """Pixels past the line along the flow; negative before the line."""
    dx, dy = _unit(flow)
    return (cx - line_pos) * dx if dx else (cy - line_pos) * dy


def step_toward_line(flow, line_pos, cx, cy, step) -> tuple:
    """(vx, vy) of `step` pixels that moves a point towards the line (used when a track has no speed yet)."""
    dx, dy = _unit(flow)
    sign = 1 if progress(flow, line_pos, cx, cy) < 0 else -1
    return dx * step * sign, dy * step * sign


def lane_overlap(flow, box_a, box_b) -> float:
    """Overlap of two boxes across the flow (same roller channel = they overlap). Negative = gap in px."""
    if line_orientation(flow) == VERTICAL:      # flow along x, lanes differ in y
        return min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])


def stack_crops(crops):
    """Put ROI crops into one image for a single YOLO call: one above the other when they are wide,
    side by side when they are tall. Returns (image, [(x, y, w, h) of each crop inside it])."""
    if not crops:
        return None, []
    wide = sum(c.shape[1] for c in crops) >= sum(c.shape[0] for c in crops)
    places, pos = [], 0
    if wide:
        width = max(c.shape[1] for c in crops)
        canvas = np.zeros((sum(c.shape[0] for c in crops), width, 3), np.uint8)
        for c in crops:
            h, w = c.shape[:2]
            canvas[pos:pos + h, 0:w] = c
            places.append((0, pos, w, h))
            pos += h
    else:
        height = max(c.shape[0] for c in crops)
        canvas = np.zeros((height, sum(c.shape[1] for c in crops), 3), np.uint8)
        for c in crops:
            h, w = c.shape[:2]
            canvas[0:h, pos:pos + w] = c
            places.append((pos, 0, w, h))
            pos += w
    return canvas, places


def place_index(places, x, y):
    """Which stacked crop contains the point, or -1."""
    for i, (px, py, w, h) in enumerate(places):
        if px <= x <= px + w and py <= y <= py + h:
            return i
    return -1


def draw_line(frame, flow, line_pos, colour, thickness):
    if line_pos is None:
        return
    if line_orientation(flow) == VERTICAL:
        cv2.line(frame, (int(line_pos), 0), (int(line_pos), frame.shape[0]), colour, thickness)
    else:
        cv2.line(frame, (0, int(line_pos)), (frame.shape[1], int(line_pos)), colour, thickness)
