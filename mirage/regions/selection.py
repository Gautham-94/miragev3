"""Region selection: decides which crop(s) of each frame get sent to the detector.

Spec reference: NVR_PIPELINE_IMPLEMENTATION_SPEC.md section 4.
"""

from __future__ import annotations

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in full detect-frame pixel coordinates

REGION_CLUSTER_MULTIPLIER = 1.35


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_inside_extent(box: Box, extent: Box) -> bool:
    """True if `box` falls fully within `extent`'s bounds (used for "is this motion box
    already covered by an existing region" checks -- e.g. skip a motion box that's fully
    inside a tracked-object's region already).
    """
    x1, y1, x2, y2 = box
    ex1, ey1, ex2, ey2 = extent
    return x1 >= ex1 and y1 >= ey1 and x2 <= ex2 and y2 <= ey2


def box_overlaps_extent(box: Box, extent: Box) -> bool:
    """True if `box` overlaps `extent` at all (used for the cluster-membership test).
    Overlap, not full containment, is the right join criterion here: a candidate box that
    partially overlaps -- or even just touches -- an existing cluster's expanded extent
    should merge into it; requiring the WHOLE candidate box to fit inside a (possibly
    still tiny, for a 1-box cluster) expanded extent is far too restrictive and would
    leave genuinely-adjacent or overlapping motion contours in separate clusters.
    """
    x1, y1, x2, y2 = box
    ex1, ey1, ex2, ey2 = extent
    return x1 < ex2 and ex1 < x2 and y1 < ey2 and ey1 < y2


def cluster_extent(boxes: list[Box], multiplier: float = REGION_CLUSTER_MULTIPLIER) -> Box:
    """The bounding extent of a cluster, expanded by `multiplier` around its center, used
    to test whether a new candidate box should join this cluster.
    """
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) / 2 * multiplier
    half_h = (y2 - y1) / 2 * multiplier
    return (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))


def cluster_boxes(boxes: list[Box], multiplier: float = REGION_CLUSTER_MULTIPLIER) -> list[list[Box]]:
    """Greedily groups nearby boxes into clusters: a box joins the first existing cluster
    whose (expanded) extent already contains it; otherwise it starts a new cluster. This
    collapses several small nearby motion contours into one crop region instead of many
    overlapping small ones (spec section 4, point 3).
    """
    clusters: list[list[Box]] = []
    for box in boxes:
        placed = False
        for cluster in clusters:
            if box_overlaps_extent(box, cluster_extent(cluster, multiplier)):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])
    return clusters


def region_from_cluster(
    cluster: list[Box], min_region_size: int, frame_shape: tuple[int, int], multiplier: float = REGION_CLUSTER_MULTIPLIER
) -> Box:
    """Computes a single square crop region for a cluster: region_size =
    max(cluster_width, cluster_height) * multiplier, rounded up to a multiple of 4, and
    floor-clamped to at least min_region_size (the model's own input size), per spec
    section 4 point 3. Centered on the cluster's centroid, clamped to frame bounds.
    """
    x1 = min(b[0] for b in cluster)
    y1 = min(b[1] for b in cluster)
    x2 = max(b[2] for b in cluster)
    y2 = max(b[3] for b in cluster)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    cluster_w, cluster_h = x2 - x1, y2 - y1
    region_size = max(cluster_w, cluster_h) * multiplier
    region_size = max(region_size, min_region_size)
    region_size = int((region_size + 3) // 4 * 4)  # round up to a multiple of 4

    half = region_size / 2
    rx1, ry1 = int(cx - half), int(cy - half)
    rx2, ry2 = rx1 + region_size, ry1 + region_size

    return _clamp_region_preserving_size(rx1, ry1, rx2, ry2, region_size, frame_shape)


def _clamp_region_preserving_size(x1: int, y1: int, x2: int, y2: int, size: int, frame_shape: tuple[int, int]) -> Box:
    """Shifts (rather than shrinks) the region to stay in-bounds, so the crop keeps its
    exact intended size whenever the frame is at least that large.
    """
    height, width = frame_shape
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        shift = x2 - width
        x1 = max(0, x1 - shift)
        x2 = width
    if y2 > height:
        shift = y2 - height
        y1 = max(0, y1 - shift)
        y2 = height
    return (x1, y1, min(x2, width), min(y2, height))


def build_regions(
    tracked_object_boxes: list[Box],
    motion_boxes: list[Box],
    min_region_size: int,
    frame_shape: tuple[int, int],
    is_calibrating: bool,
    ptz_moving: bool = False,
) -> list[Box]:
    """Spec section 4, points 1-3: builds the full list of regions to run the detector on
    this frame. Tracked-object regions are always included; motion-triggered regions are
    only added when the motion detector isn't calibrating and no PTZ move is in progress.
    """
    regions: list[Box] = []

    for cluster in cluster_boxes(tracked_object_boxes):
        regions.append(region_from_cluster(cluster, min_region_size, frame_shape))

    if not is_calibrating and not ptz_moving:
        uncovered_motion = [
            box for box in motion_boxes
            if not any(box_inside_extent(box, r) for r in regions)
        ]
        for cluster in cluster_boxes(uncovered_motion):
            regions.append(region_from_cluster(cluster, min_region_size, frame_shape))

    return regions


def denormalize_box(normalized_box: tuple[float, float, float, float], region: Box) -> Box:
    """Converts a region-relative normalized box [y1,x1,y2,x2] (as returned by
    RemoteObjectDetector.detect(), spec section 3.2.2) back into full detect-frame pixel
    coordinates, per spec section 4's "denormalize each region-relative box" step.
    """
    y1, x1, y2, x2 = normalized_box
    rx1, ry1, rx2, ry2 = region
    region_w = rx2 - rx1
    region_h = ry2 - ry1
    return (
        int(rx1 + x1 * region_w),
        int(ry1 + y1 * region_h),
        int(rx1 + x2 * region_w),
        int(ry1 + y2 * region_h),
    )
