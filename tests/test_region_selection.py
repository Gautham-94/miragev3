from __future__ import annotations

from mirage.regions.selection import (
    box_inside_extent,
    build_regions,
    cluster_boxes,
    denormalize_box,
    region_from_cluster,
)


def test_cluster_boxes_merges_overlapping_boxes():
    # Two overlapping boxes must merge into one cluster (any positive overlap easily
    # falls within the 1.35x-expanded extent test).
    boxes = [(10, 10, 20, 20), (15, 12, 25, 22)]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 1
    assert set(clusters[0]) == set(boxes)


def test_cluster_boxes_does_not_merge_boxes_just_outside_expanded_extent():
    # A single small box's cluster_extent (1.35x expansion around its own center) only
    # grows a 10x10 box by ~1-2px per side -- a second box starting just past that
    # expanded boundary should NOT be merged. This documents the real (tight, not a loose
    # "nearby" heuristic) clustering behavior from spec section 4 point 3.
    boxes = [(10, 10, 20, 20), (22, 12, 32, 22)]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 2


def test_cluster_boxes_keeps_far_apart_boxes_separate():
    boxes = [(0, 0, 10, 10), (500, 500, 520, 520)]
    clusters = cluster_boxes(boxes)
    assert len(clusters) == 2


def test_cluster_boxes_empty_input():
    assert cluster_boxes([]) == []


def test_region_from_cluster_respects_min_size():
    cluster = [(100, 100, 110, 110)]  # tiny 10x10 box
    region = region_from_cluster(cluster, min_region_size=320, frame_shape=(720, 1280))
    x1, y1, x2, y2 = region
    assert (x2 - x1) >= 320
    assert (y2 - y1) >= 320


def test_region_from_cluster_size_is_multiple_of_4():
    cluster = [(0, 0, 47, 47)]
    region = region_from_cluster(cluster, min_region_size=1, frame_shape=(720, 1280))
    x1, y1, x2, y2 = region
    assert (x2 - x1) % 4 == 0


def test_region_from_cluster_clamped_to_frame_bounds_preserving_size():
    # Cluster near the top-left corner -- region would extend past (0,0) without clamping.
    cluster = [(0, 0, 20, 20)]
    frame_shape = (480, 640)
    region = region_from_cluster(cluster, min_region_size=320, frame_shape=frame_shape)
    x1, y1, x2, y2 = region
    assert x1 >= 0 and y1 >= 0
    assert x2 <= frame_shape[1] and y2 <= frame_shape[0]
    # Size should be preserved (not shrunk) via the shift-not-shrink clamp, since the
    # frame is large enough to accommodate a 320px region.
    assert (x2 - x1) == 320
    assert (y2 - y1) == 320


def test_region_from_cluster_clamped_when_frame_smaller_than_region():
    cluster = [(0, 0, 10, 10)]
    frame_shape = (100, 100)  # smaller than the requested 320 min region size
    region = region_from_cluster(cluster, min_region_size=320, frame_shape=frame_shape)
    x1, y1, x2, y2 = region
    assert x1 == 0 and y1 == 0
    assert x2 == 100 and y2 == 100


def test_build_regions_includes_tracked_object_regions_always():
    tracked_boxes = [(50, 50, 100, 100)]
    regions = build_regions(tracked_boxes, motion_boxes=[], min_region_size=64, frame_shape=(480, 640),
                             is_calibrating=True, ptz_moving=False)
    assert len(regions) == 1


def test_build_regions_suppresses_motion_regions_while_calibrating():
    motion_boxes = [(200, 200, 250, 250)]
    regions = build_regions([], motion_boxes, min_region_size=64, frame_shape=(480, 640),
                             is_calibrating=True, ptz_moving=False)
    assert regions == []


def test_build_regions_suppresses_motion_regions_during_ptz_move():
    motion_boxes = [(200, 200, 250, 250)]
    regions = build_regions([], motion_boxes, min_region_size=64, frame_shape=(480, 640),
                             is_calibrating=False, ptz_moving=True)
    assert regions == []


def test_build_regions_includes_motion_regions_when_not_calibrating_and_no_ptz():
    motion_boxes = [(200, 200, 250, 250)]
    regions = build_regions([], motion_boxes, min_region_size=64, frame_shape=(480, 640),
                             is_calibrating=False, ptz_moving=False)
    assert len(regions) == 1


def test_build_regions_motion_box_already_covered_by_tracked_region_not_duplicated():
    tracked_boxes = [(100, 100, 200, 200)]
    # Motion box fully inside the tracked-object's resulting region -- must not add a
    # second, redundant region for it.
    motion_boxes = [(140, 140, 160, 160)]
    regions = build_regions(tracked_boxes, motion_boxes, min_region_size=64, frame_shape=(480, 640),
                             is_calibrating=False, ptz_moving=False)
    assert len(regions) == 1


def test_denormalize_box_maps_region_relative_to_frame_coords():
    region = (100, 100, 420, 420)  # 320x320 region
    # normalized box covering the left half of the region: y1,x1,y2,x2
    normalized = (0.0, 0.0, 1.0, 0.5)
    box = denormalize_box(normalized, region)
    x1, y1, x2, y2 = box
    assert x1 == 100
    assert y1 == 100
    assert x2 == 100 + int(0.5 * 320)
    assert y2 == 420


def test_denormalize_box_full_region():
    region = (10, 20, 110, 220)
    normalized = (0.0, 0.0, 1.0, 1.0)
    assert denormalize_box(normalized, region) == (10, 20, 110, 220)


def test_box_inside_extent():
    assert box_inside_extent((10, 10, 20, 20), (0, 0, 100, 100)) is True
    assert box_inside_extent((10, 10, 120, 20), (0, 0, 100, 100)) is False
