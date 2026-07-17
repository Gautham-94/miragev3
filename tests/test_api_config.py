"""Tests for the config-write endpoints (mirage/api/routers/config.py) -- the add-camera
wizard's actual save path. Uses create_app(db_path=...) (config=None, the real DB-backed
mode) since these endpoints only make sense against a real, mutable config store.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mirage.api.app import create_app
from mirage.config.schema import MirageConfig


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        app = create_app(db_path=db_path)
        with TestClient(app) as c:
            yield c


def test_list_detectors_returns_default_general_detector(client):
    resp = client.get("/api/config/detectors")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "general"
    assert data[0]["device"] == "onnx_yolov8"
    assert data[0]["execution_provider"] == "cpu"  # schema default, unless configured otherwise


def test_list_execution_providers_always_includes_cpu_and_auto(client):
    resp = client.get("/api/config/execution-providers")
    assert resp.status_code == 200
    providers = resp.json()
    assert "cpu" in providers
    assert "auto" in providers  # always resolvable, degrades to cpu with no accelerator present


def test_create_detector_with_explicit_cpu_execution_provider(client):
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "cpu",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["execution_provider"] == "cpu"


def test_create_detector_with_unavailable_execution_provider_returns_422(client):
    # cuda is a real ExecutionProvider value, but this dev/test machine has no NVIDIA
    # GPU -- the endpoint must reject configuring a detector for hardware that isn't
    # actually present, rather than silently accepting it and failing later at
    # pipeline-startup InferenceSession-construction time.
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "cuda",
        },
    )
    assert resp.status_code == 422


def test_create_camera_succeeds_and_is_visible_immediately(client):
    resp = client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://127.0.0.1/front_door",
            "detector": "general",
            "track_objects": ["person", "car"],
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ok"] is True
    assert body["restart_required"] is True
    assert body["camera"]["name"] == "front_door"
    assert body["camera"]["track_objects"] == ["person", "car"]

    listing = client.get("/api/cameras")
    names = [c["name"] for c in listing.json()]
    assert "front_door" in names


def test_create_camera_duplicate_name_returns_409(client):
    payload = {
        "name": "front_door",
        "rtsp_url": "rtsp://127.0.0.1/front_door",
        "detector": "general",
    }
    first = client.post("/api/config/cameras", json=payload)
    assert first.status_code == 201

    second = client.post("/api/config/cameras", json=payload)
    assert second.status_code == 409


def test_create_camera_unknown_detector_returns_422(client):
    resp = client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "nonexistent"},
    )
    assert resp.status_code == 422


def test_create_camera_missing_required_field_returns_422(client):
    resp = client.post("/api/config/cameras", json={"name": "front_door", "detector": "general"})
    assert resp.status_code == 422


def test_update_camera_changes_persist(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general", "fps": 5},
    )

    resp = client.put(
        "/api/config/cameras/front_door",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general", "fps": 15},
    )
    assert resp.status_code == 200
    assert resp.json()["camera"]["fps"] == 15

    fetched = client.get("/api/cameras/front_door")
    assert fetched.json()["fps"] == 15


def test_create_camera_track_all_defaults_to_false(client):
    resp = client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general"},
    )
    assert resp.status_code == 201
    assert resp.json()["camera"]["track_all"] is False

    fetched = client.get("/api/cameras/front_door")
    assert fetched.json()["track_all"] is False


def test_create_camera_with_track_all_persists_and_is_readable(client):
    resp = client.post(
        "/api/config/cameras",
        json={
            "name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general",
            "track_objects": ["person"], "track_all": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["camera"]["track_all"] is True

    fetched_summary = client.get("/api/cameras/front_door")
    assert fetched_summary.json()["track_all"] is True

    fetched_config = client.get("/api/config/cameras/front_door")
    assert fetched_config.json()["track_all"] is True
    # track_objects itself is untouched/still saved, even though track_all makes it
    # irrelevant to the tracking gate -- so re-disabling track_all later doesn't lose it.
    assert fetched_config.json()["track_objects"] == ["person"]


def test_update_camera_can_toggle_track_all_on_and_back_off(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general", "track_objects": ["person"]},
    )

    turned_on = client.put(
        "/api/config/cameras/front_door",
        json={
            "name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general",
            "track_objects": ["person"], "track_all": True,
        },
    )
    assert turned_on.json()["camera"]["track_all"] is True

    turned_off = client.put(
        "/api/config/cameras/front_door",
        json={
            "name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general",
            "track_objects": ["person"], "track_all": False,
        },
    )
    assert turned_off.json()["camera"]["track_all"] is False
    # track_objects survived the round trip through track_all=True unchanged.
    assert turned_off.json()["camera"]["track_objects"] == ["person"]


def test_update_unknown_camera_returns_404(client):
    resp = client.put(
        "/api/config/cameras/nonexistent",
        json={"name": "nonexistent", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general"},
    )
    assert resp.status_code == 404


def test_delete_camera_removes_it(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/x", "detector": "general"},
    )

    resp = client.delete("/api/config/cameras/front_door")
    assert resp.status_code == 204

    listing = client.get("/api/cameras")
    assert listing.json() == []


def test_delete_unknown_camera_returns_404(client):
    resp = client.delete("/api/config/cameras/nonexistent")
    assert resp.status_code == 404


def test_create_camera_with_udp_transport_persists_and_is_readable(client):
    # Regression coverage for the real Hikvision-camera incident this option exists to
    # fix (see CameraInputConfig.rtsp_transport's docstring) -- a camera whose
    # RTSP-over-TCP session gets reset almost immediately needs a per-camera way to opt
    # into UDP transport instead, settable through the same wizard/API path as everything
    # else.
    resp = client.post(
        "/api/config/cameras",
        json={
            "name": "flaky_tcp_cam",
            "rtsp_url": "rtsp://admin:pass@192.168.1.99:554/stream1",
            "detector": "general",
            "rtsp_transport": "udp",
        },
    )
    assert resp.status_code == 201

    config = MirageConfig.from_db()
    assert config.cameras["flaky_tcp_cam"].ffmpeg.inputs[0].rtsp_transport.value == "udp"


def test_list_camera_configs_includes_rtsp_url_and_transport(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://admin:pass@192.168.1.99:554/stream1",
            "detector": "general",
            "rtsp_transport": "udp",
        },
    )

    resp = client.get("/api/config/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert len(cameras) == 1
    assert cameras[0]["name"] == "front_door"
    assert cameras[0]["rtsp_url"] == "rtsp://admin:pass@192.168.1.99:554/stream1"
    assert cameras[0]["rtsp_transport"] == "udp"


def test_get_camera_config_by_name_for_edit_prefill(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://admin:pass@192.168.1.99:554/stream1",
            "detector": "general",
            "track_objects": ["person", "car"],
            "fps": 15,
            "retain_days": 3,
            "alert_labels": ["person"],
        },
    )

    resp = client.get("/api/config/cameras/front_door")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rtsp_url"] == "rtsp://admin:pass@192.168.1.99:554/stream1"
    assert body["rtsp_transport"] == "tcp"
    assert body["track_objects"] == ["person", "car"]
    assert body["fps"] == 15
    assert body["retain_days"] == 3
    assert body["alert_labels"] == ["person"]


def test_get_camera_config_unknown_returns_404(client):
    resp = client.get("/api/config/cameras/nonexistent")
    assert resp.status_code == 404


def test_general_cameras_endpoint_does_not_leak_rtsp_credentials(client):
    # GET /api/cameras (mirage/api/routers/cameras.py) is the general-purpose,
    # broadly-consumed listing (e.g. the Live page's camera grid) and must NOT expose
    # rtsp_url, since that would leak embedded camera credentials to every reader --
    # only the dedicated GET /api/config/cameras (config-write router, used for the
    # edit-form pre-fill) returns it.
    client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://admin:supersecret@192.168.1.99:554/stream1",
            "detector": "general",
        },
    )

    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    body = resp.json()
    assert "rtsp_url" not in body[0]
    assert "supersecret" not in str(body)


def test_create_detector_succeeds_and_is_visible_immediately(client):
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "width": 416,
            "height": 416,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "wildlife_detector"
    assert body["device"] == "onnx_yolov8"
    assert body["model_width"] == 416
    assert body["model_height"] == 416

    listing = client.get("/api/config/detectors")
    names = [d["name"] for d in listing.json()]
    assert "wildlife_detector" in names


def test_create_detector_duplicate_name_returns_409(client):
    payload = {
        "name": "wildlife_detector",
        "model_path": "models/yolov8n.onnx",
        "labelmap_path": "models/coco_labelmap.txt",
    }
    first = client.post("/api/config/detectors", json=payload)
    assert first.status_code == 201

    second = client.post("/api/config/detectors", json=payload)
    assert second.status_code == 409


def test_create_detector_unknown_backend_returns_422(client):
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "device": "nonexistent_backend",
        },
    )
    assert resp.status_code == 422


def test_create_detector_missing_model_path_returns_422(client):
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/does_not_exist.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
        },
    )
    assert resp.status_code == 422


def test_create_detector_missing_labelmap_path_returns_422(client):
    resp = client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/does_not_exist.txt",
        },
    )
    assert resp.status_code == 422


def test_delete_detector_removes_it(client):
    client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
        },
    )

    resp = client.delete("/api/config/detectors/wildlife_detector")
    assert resp.status_code == 204

    listing = client.get("/api/config/detectors")
    names = [d["name"] for d in listing.json()]
    assert "wildlife_detector" not in names


def test_delete_unknown_detector_returns_404(client):
    resp = client.delete("/api/config/detectors/nonexistent")
    assert resp.status_code == 404


def test_delete_detector_still_referenced_by_camera_returns_409(client):
    client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
        },
    )
    client.post(
        "/api/config/cameras",
        json={"name": "backyard", "rtsp_url": "rtsp://127.0.0.1/backyard", "detector": "wildlife_detector"},
    )

    resp = client.delete("/api/config/detectors/wildlife_detector")
    assert resp.status_code == 409

    # still there, unaffected by the rejected delete
    listing = client.get("/api/config/detectors")
    names = [d["name"] for d in listing.json()]
    assert "wildlife_detector" in names


def test_update_detector_changes_execution_provider(client):
    # The scenario this endpoint exists for: a detector was registered on CPU, and the
    # user later wants to switch it to a faster accelerator once they've confirmed it
    # works, without deleting/recreating (which cameras referencing it would block).
    client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "cpu",
        },
    )

    resp = client.put(
        "/api/config/detectors/wildlife_detector",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "auto",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["execution_provider"] == "auto"

    listing = client.get("/api/config/detectors")
    detector = next(d for d in listing.json() if d["name"] == "wildlife_detector")
    assert detector["execution_provider"] == "auto"


def test_update_detector_while_camera_references_it_succeeds(client):
    # Unlike delete, editing in place must NOT be blocked by existing camera references
    # -- that's the whole point of this endpoint over delete+recreate.
    client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "cpu",
        },
    )
    client.post(
        "/api/config/cameras",
        json={"name": "backyard", "rtsp_url": "rtsp://127.0.0.1/backyard", "detector": "wildlife_detector"},
    )

    resp = client.put(
        "/api/config/detectors/wildlife_detector",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "auto",
        },
    )
    assert resp.status_code == 200

    # the camera's reference is unaffected -- still points at the same (now-updated) detector
    camera = client.get("/api/config/cameras/backyard").json()
    assert camera["detector"] == "wildlife_detector"


def test_update_unknown_detector_returns_404(client):
    resp = client.put(
        "/api/config/detectors/nonexistent",
        json={"name": "nonexistent", "model_path": "models/yolov8n.onnx", "labelmap_path": "models/coco_labelmap.txt"},
    )
    assert resp.status_code == 404


def test_update_detector_rename_returns_422(client):
    client.post(
        "/api/config/detectors",
        json={"name": "wildlife_detector", "model_path": "models/yolov8n.onnx", "labelmap_path": "models/coco_labelmap.txt"},
    )

    resp = client.put(
        "/api/config/detectors/wildlife_detector",
        json={"name": "renamed_detector", "model_path": "models/yolov8n.onnx", "labelmap_path": "models/coco_labelmap.txt"},
    )
    assert resp.status_code == 422


def test_update_detector_unavailable_execution_provider_returns_422(client):
    client.post(
        "/api/config/detectors",
        json={"name": "wildlife_detector", "model_path": "models/yolov8n.onnx", "labelmap_path": "models/coco_labelmap.txt"},
    )

    resp = client.put(
        "/api/config/detectors/wildlife_detector",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "execution_provider": "cuda",
        },
    )
    assert resp.status_code == 422


def test_created_detector_config_is_actually_valid_mirage_config(client):
    # End-to-end proof this isn't just writing arbitrary JSON: the saved config must
    # round-trip through MirageConfig's own Pydantic validation cleanly, same as it
    # would if MirageApp loaded it on the next restart.
    client.post(
        "/api/config/detectors",
        json={
            "name": "wildlife_detector",
            "model_path": "models/yolov8n.onnx",
            "labelmap_path": "models/coco_labelmap.txt",
            "width": 416,
            "height": 416,
        },
    )

    config = MirageConfig.from_db()
    detector = config.detectors["wildlife_detector"]
    assert detector.device == "onnx_yolov8"
    assert detector.model.width == 416
    assert detector.model.height == 416
    assert detector.model.model_path == "models/yolov8n.onnx"
    assert detector.model.labelmap_path == "models/coco_labelmap.txt"


def test_created_camera_config_is_actually_valid_mirage_config(client):
    # End-to-end proof this isn't just writing arbitrary JSON: the saved config must
    # round-trip through MirageConfig's own Pydantic validation cleanly, same as it
    # would if MirageApp loaded it on the next restart.
    client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://127.0.0.1/front_door",
            "detector": "general",
            "width": 1280,
            "height": 720,
            "fps": 10,
            "alert_labels": ["person"],
        },
    )

    config = MirageConfig.from_db()
    camera = config.cameras["front_door"]
    assert camera.detect.width == 1280
    assert camera.detect.height == 720
    assert camera.review.alerts.labels == ["person"]
    assert camera.ffmpeg.inputs[0].path == "rtsp://127.0.0.1/front_door"


def test_camera_openvocab_direct_frame_defaults_to_false(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/front_door", "detector": "general"},
    )
    config = MirageConfig.from_db()
    assert config.cameras["front_door"].openvocab_direct_frame is False

    listing = client.get("/api/config/cameras").json()
    assert listing[0]["openvocab_direct_frame"] is False


def test_camera_openvocab_direct_frame_can_be_enabled(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "front_door",
            "rtsp_url": "rtsp://127.0.0.1/front_door",
            "detector": "general",
            "openvocab_direct_frame": True,
        },
    )
    config = MirageConfig.from_db()
    assert config.cameras["front_door"].openvocab_direct_frame is True

    fetched = client.get("/api/config/cameras/front_door").json()
    assert fetched["openvocab_direct_frame"] is True


def test_track_objects_word_not_in_labelmap_auto_provisions_a_query(client):
    """The core track_objects/open-vocab bridge: typing a word the routed detector's
    labelmap doesn't know (general's is models/coco_labelmap.txt, no generic "animal"
    class -- TODO_FIX_LIST.md item 4) into Track objects must auto-create a scoped,
    enabled OpenVocabQuery for it, tagged source="track_objects" -- not leave it as a
    silently dead config value the way it used to be.
    """
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["person", "animals"],
        },
    )

    queries = client.get("/api/config/queries").json()
    assert len(queries) == 1
    assert queries[0]["text"] == "animals"
    assert queries[0]["cameras"] == ["garage"]
    assert queries[0]["enabled"] is True
    assert queries[0]["source"] == "track_objects"

    # "person" IS a real COCO label -- it must NOT also get an auto-provisioned query.
    assert all(q["text"] != "person" for q in queries)


def test_track_objects_auto_provisioning_enables_direct_frame_mode(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    config = MirageConfig.from_db()
    assert config.cameras["garage"].openvocab_direct_frame is True


def test_track_objects_auto_provisioning_adds_alert_label(client):
    """So a real OWLv2 match for "animals" actually drives ReviewSegment severity the
    same way a native COCO word like "person" already does -- otherwise the match would
    persist as a QueryMatch row but never surface as a review/alert at all.
    """
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    config = MirageConfig.from_db()
    assert "animals" in config.cameras["garage"].review.alerts.labels


def test_removing_word_from_track_objects_deletes_its_auto_query(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    assert len(client.get("/api/config/queries").json()) == 1

    client.put(
        "/api/config/cameras/garage",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["person"],
        },
    )
    assert client.get("/api/config/queries").json() == []


def test_deleting_camera_removes_its_auto_provisioned_queries(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    assert len(client.get("/api/config/queries").json()) == 1

    client.delete("/api/config/cameras/garage")
    assert client.get("/api/config/queries").json() == []


def test_manual_query_untouched_by_track_objects_sync_on_other_camera(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/front_door", "detector": "general"},
    )
    manual = client.post(
        "/api/config/queries", json={"text": "a red backpack", "cameras": ["front_door"]}
    ).json()
    assert manual["source"] == "manual"

    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )

    queries = client.get("/api/config/queries").json()
    assert len(queries) == 2
    manual_after = next(q for q in queries if q["id"] == manual["id"])
    assert manual_after["text"] == "a red backpack"
    assert manual_after["source"] == "manual"


def test_auto_provisioned_query_cannot_be_edited_directly(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    auto_query = client.get("/api/config/queries").json()[0]

    resp = client.put(f"/api/config/queries/{auto_query['id']}", json={"text": "something else"})
    assert resp.status_code == 409


def test_auto_provisioned_query_cannot_be_deleted_directly(client):
    client.post(
        "/api/config/cameras",
        json={
            "name": "garage",
            "rtsp_url": "rtsp://127.0.0.1/garage",
            "detector": "general",
            "track_objects": ["animals"],
        },
    )
    auto_query = client.get("/api/config/queries").json()[0]

    resp = client.delete(f"/api/config/queries/{auto_query['id']}")
    assert resp.status_code == 409
    assert len(client.get("/api/config/queries").json()) == 1


def test_list_queries_returns_empty_by_default(client):
    resp = client.get("/api/config/queries")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_query_succeeds_and_is_visible_immediately(client):
    resp = client.post("/api/config/queries", json={"text": "a dog off-leash"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["text"] == "a dog off-leash"
    assert body["cameras"] == []
    assert body["enabled"] is True
    assert body["id"]  # a real id was assigned

    listing = client.get("/api/config/queries")
    assert len(listing.json()) == 1
    assert listing.json()[0]["text"] == "a dog off-leash"


def test_create_query_scoped_to_unknown_camera_returns_422(client):
    resp = client.post("/api/config/queries", json={"text": "a dog off-leash", "cameras": ["nonexistent"]})
    assert resp.status_code == 422


def test_create_query_scoped_to_real_camera_succeeds(client):
    client.post(
        "/api/config/cameras",
        json={"name": "front_door", "rtsp_url": "rtsp://127.0.0.1/front_door", "detector": "general"},
    )
    resp = client.post("/api/config/queries", json={"text": "a red backpack", "cameras": ["front_door"]})
    assert resp.status_code == 201
    assert resp.json()["cameras"] == ["front_door"]


def test_update_query_changes_text_and_enabled(client):
    created = client.post("/api/config/queries", json={"text": "a dog off-leash"}).json()

    resp = client.put(
        f"/api/config/queries/{created['id']}",
        json={"text": "a cat on the fence", "enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "a cat on the fence"
    assert resp.json()["enabled"] is False
    assert resp.json()["id"] == created["id"]  # id is preserved across an edit

    listing = client.get("/api/config/queries")
    assert len(listing.json()) == 1
    assert listing.json()[0]["text"] == "a cat on the fence"


def test_update_unknown_query_returns_404(client):
    resp = client.put("/api/config/queries/nonexistent", json={"text": "anything"})
    assert resp.status_code == 404


def test_delete_query_removes_it(client):
    created = client.post("/api/config/queries", json={"text": "a dog off-leash"}).json()

    resp = client.delete(f"/api/config/queries/{created['id']}")
    assert resp.status_code == 204

    assert client.get("/api/config/queries").json() == []


def test_delete_unknown_query_returns_404(client):
    resp = client.delete("/api/config/queries/nonexistent")
    assert resp.status_code == 404


def test_created_query_config_is_actually_valid_mirage_config(client):
    client.post("/api/config/queries", json={"text": "a dog off-leash"})

    config = MirageConfig.from_db()
    assert len(config.queries) == 1
    assert config.queries[0].text == "a dog off-leash"
