import threading
from pathlib import Path

import pytest

from sscraper.recognition import FaceMatcher, NoFace


class FakeDeepFace:
    def __init__(self):
        self.calls = []

    def represent(self, img_path, **kwargs):
        self.calls.append(img_path)
        assert kwargs["enforce_detection"] is True
        if img_path == "blank":
            raise ValueError("Face could not be detected")
        if img_path == "broken":
            raise ValueError("Model file is broken")
        embeddings = {
            "reference1": [[1.0]],
            "reference2": [[1.1]],
            "candidate": [[0.0], [1.05]],
            "group": [[0.0], [1.0]],
        }[img_path]
        return [
            {"embedding": e, "facial_area": {"x": index * 10, "y": 0, "w": 10, "h": 10}}
            for index, e in enumerate(embeddings)
        ]

    def verify(self, img1_path, img2_path, threshold, **_):
        distance = abs(img1_path[0] - img2_path[0])
        threshold = threshold if threshold is not None else 0.2
        return {"distance": distance, "threshold": threshold, "verified": distance <= threshold}


def test_references_are_cached_and_all_candidate_faces_are_compared():
    api = FakeDeepFace()
    matcher = FaceMatcher("SFace", "opencv", None, threading.Event(), api)
    matcher.prepare([Path("reference1"), Path("reference2")], lambda *_: None)
    match = matcher.compare(Path("candidate"))
    assert match.matched
    assert match.supporting_references == 2
    assert match.face_count == 2
    assert match.face_area["x"] == 10
    assert api.calls == ["reference1", "reference2", "candidate"]


def test_multiple_face_reference_rejected():
    matcher = FaceMatcher("SFace", "opencv", None, threading.Event(), FakeDeepFace())
    with pytest.raises(ValueError, match="Crop"):
        matcher.prepare([Path("group")], lambda *_: None)


def test_no_face_and_model_errors_are_distinct():
    matcher = FaceMatcher("SFace", "opencv", None, threading.Event(), FakeDeepFace())
    with pytest.raises(NoFace):
        matcher.represent(Path("blank"))
    with pytest.raises(ValueError, match="Model file"):
        matcher.represent(Path("broken"))


def test_threshold_override_is_used():
    matcher = FaceMatcher("SFace", "opencv", 0.001, threading.Event(), FakeDeepFace())
    matcher.prepare([Path("reference1")], lambda *_: None)
    assert not matcher.compare(Path("candidate")).matched
