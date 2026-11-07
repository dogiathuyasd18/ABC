import numpy as np
import pytest

from src.prep.extraction import _find_head_device, _frames_to_2d, extract_head_pose


def test_head_offset_skips_controller_axes():
    devices = [
        {"name": "left", "joint": "HAND", "type": "controller", "axes": ["i", "j", "k", "1"]},
        {"name": "hmd", "joint": "HEAD", "type": "hmd", "axes": ["i", "j", "k", "1"]},
    ]
    index, device, offset = _find_head_device(devices)
    assert index == 1
    assert device["joint"] == "HEAD"
    assert offset == 1 + 4


def test_rejects_missing_head():
    with pytest.raises(ValueError, match="HEAD"):
        _find_head_device([{"name": "hand", "joint": "HAND", "axes": ["i", "j", "k", "1"]}])


def test_frames_reshape_and_extract_without_fpzip(monkeypatch):
    devices = [
        {"name": "hmd", "joint": "HEAD", "type": "hmd", "axes": ["1", "i", "j", "k"]},
    ]
    n = 5
    frames = np.zeros((1, n, 5), dtype=np.float32)
    frames[0, :, 0] = np.arange(n) * 0.01
    # stored axes 1,i,j,k in columns 1:5 → we request i,j,k,1
    frames[0, :, 1] = 1.0
    frames[0, :, 2] = 0.1
    frames[0, :, 3] = 0.2
    frames[0, :, 4] = 0.3

    def fake_decompress(payload, order="C"):
        return frames

    monkeypatch.setattr("src.prep.extraction.fpzip.decompress", fake_decompress)
    document = {
        "$schema": "test",
        "frames": b"unused",
        "info": {"user": {"id": "u"}, "hardware": {"devices": devices}},
    }
    extracted = extract_head_pose(document)
    assert extracted.quaternions.shape == (n, 4)
    assert extracted.schema["verified_head_not_controller"]
    assert extracted.schema["head_device_index"] == 0
    assert extracted.schema["quaternion_columns"] == [1 + 1, 1 + 2, 1 + 3, 1 + 0]
    np.testing.assert_allclose(extracted.quaternions[0], [0.1, 0.2, 0.3, 1.0])
    assert extracted.schema["frame_shape_original"] == [1, n, 5]
