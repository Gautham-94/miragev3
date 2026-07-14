from __future__ import annotations

from mirage.regions.reduce import (
    RawDetection,
    box_area,
    get_consolidated_object_detections,
    intersection_area,
    is_clipped_at_region_edge,
    reduce_detections,
    reduce_overlapping_detections,
)

FRAME_SHAPE = (480, 640)  # height, width


def test_box_area():
    assert box_area((0, 0, 10, 10)) == 100
    assert box_area((5, 5, 5, 5)) == 0


def test_intersection_area_overlapping():
    assert intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25


def test_intersection_area_disjoint():
    assert intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_is_clipped_at_region_edge_true_when_touching_non_frame_boundary():
    region = (100, 100, 400, 400)  # interior region, not touching frame edges
    box = (100, 200, 150, 250)  # touches region's left edge (x1=100=region x1)
    assert is_clipped_at_region_edge(box, region, FRAME_SHAPE) is True


def test_is_clipped_at_region_edge_false_when_region_edge_is_also_frame_edge():
    region = (0, 0, 300, 300)  # region's left/top edges ARE the frame edges
    box = (0, 100, 50, 150)  # touches region's left edge, which is also x=0 (frame edge)
    assert is_clipped_at_region_edge(box, region, FRAME_SHAPE) is False


def test_is_clipped_at_region_edge_false_for_interior_box():
    region = (100, 100, 400, 400)
    box = (200, 200, 250, 250)  # well inside the region, not touching any edge
    assert is_clipped_at_region_edge(box, region, FRAME_SHAPE) is False


def test_reduce_overlapping_detections_nms_collapses_duplicates():
    region = (0, 0, 640, 480)
    detections = [
        RawDetection(label="person", score=0.9, box=(100, 100, 200, 300), region=region),
        RawDetection(label="person", score=0.85, box=(102, 101, 199, 298), region=region),  # near-duplicate
    ]
    result = reduce_overlapping_detections(detections, FRAME_SHAPE)
    assert len(result) == 1
    assert result[0].score == 0.9


def test_reduce_overlapping_detections_keeps_different_labels_separate():
    region = (0, 0, 640, 480)
    detections = [
        RawDetection(label="person", score=0.9, box=(100, 100, 200, 300), region=region),
        RawDetection(label="car", score=0.8, box=(100, 100, 200, 300), region=region),  # same box, different label
    ]
    result = reduce_overlapping_detections(detections, FRAME_SHAPE)
    assert len(result) == 2
    assert {d.label for d in result} == {"person", "car"}


def test_reduce_overlapping_detections_keeps_far_apart_same_label_boxes():
    region = (0, 0, 640, 480)
    detections = [
        RawDetection(label="person", score=0.9, box=(10, 10, 60, 60), region=region),
        RawDetection(label="person", score=0.8, box=(400, 400, 450, 450), region=region),
    ]
    result = reduce_overlapping_detections(detections, FRAME_SHAPE)
    assert len(result) == 2


def test_reduce_overlapping_detections_edge_clipped_low_score_survives_nms():
    # A low-confidence detection clipped at a non-frame region edge should get its
    # confidence floored to 0.6 before NMS runs, per spec section 4.
    region = (100, 100, 400, 400)
    detections = [
        RawDetection(label="person", score=0.3, box=(100, 200, 150, 300), region=region),  # touches region's left edge
    ]
    result = reduce_overlapping_detections(detections, FRAME_SHAPE)
    # NMS_SCORE_THRESHOLD is 0.5; a raw 0.3 score would normally be dropped, but the
    # edge-clip floor (0.6) should let it survive.
    assert len(result) == 1


def test_reduce_overlapping_detections_non_clipped_low_score_dropped():
    region = (100, 100, 400, 400)
    detections = [
        RawDetection(label="person", score=0.3, box=(200, 200, 250, 250), region=region),  # interior, not clipped
    ]
    result = reduce_overlapping_detections(detections, FRAME_SHAPE)
    assert len(result) == 0


def test_get_consolidated_object_detections_drops_smaller_contained_box():
    region = (0, 0, 640, 480)
    big = RawDetection(label="car", score=0.9, box=(50, 50, 250, 250), region=region)
    small_inside_big = RawDetection(label="car", score=0.85, box=(100, 100, 150, 150), region=region)
    result = get_consolidated_object_detections([big, small_inside_big])
    assert len(result) == 1
    assert result[0] is big


def test_get_consolidated_object_detections_keeps_tiny_box_relative_to_large():
    region = (0, 0, 640, 480)
    big = RawDetection(label="car", score=0.9, box=(0, 0, 600, 400), region=region)  # huge box
    # tiny box, <5% of big's area, positioned elsewhere (not really "contained" logically,
    # simulating "a car far down the street behind a big foreground vehicle").
    tiny = RawDetection(label="car", score=0.8, box=(590, 390, 598, 398), region=region)
    result = get_consolidated_object_detections([big, tiny])
    assert len(result) == 2


def test_get_consolidated_object_detections_different_labels_unaffected():
    region = (0, 0, 640, 480)
    car = RawDetection(label="car", score=0.9, box=(50, 50, 250, 250), region=region)
    person = RawDetection(label="person", score=0.85, box=(100, 100, 150, 150), region=region)
    result = get_consolidated_object_detections([car, person])
    assert len(result) == 2


def test_reduce_detections_full_pipeline():
    region = (0, 0, 640, 480)
    detections = [
        RawDetection(label="person", score=0.9, box=(100, 100, 200, 300), region=region),
        RawDetection(label="person", score=0.85, box=(102, 101, 199, 298), region=region),  # NMS duplicate
        RawDetection(label="car", score=0.7, box=(300, 300, 400, 400), region=region),
    ]
    result = reduce_detections(detections, FRAME_SHAPE)
    labels = sorted(d.label for d in result)
    assert labels == ["car", "person"]
