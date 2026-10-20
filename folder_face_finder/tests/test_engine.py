import csv
import hashlib
import json
import threading

import pytest
from PIL import Image

from face_finder.engine import (
    Cancelled,
    Config,
    FolderScan,
    Matcher,
    NoFace,
    Result,
    collect_images,
)


def photo(path, value=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 30), (value, value, value)).save(path)
    return path


def test_discovery_recursion_excludes_references_outputs_and_symlinks(tmp_path):
    source = tmp_path / "photos"
    first = photo(source / "A.JPG")
    second = photo(source / "nested" / "second.png")
    reference = photo(source / "reference.png")
    output = source / "reports"
    photo(output / "copied.png")
    old_run = source / "old-scan"
    photo(old_run / "match.png")
    (old_run / ".folder-face-run").touch()
    (source / "link.jpg").symlink_to(first)
    (source / "linked-folder").symlink_to(source / "nested", target_is_directory=True)
    (source / "note.txt").write_text("not an image")
    config = Config(source, output, [reference])
    assert collect_images(config, threading.Event(), lambda *_: None) == [first, second]
    assert collect_images(
        Config(source, output, [reference], recursive=False), threading.Event(), lambda *_: None
    ) == [first]


def test_output_equal_source_excludes_only_marked_runs(tmp_path):
    original = photo(tmp_path / "photo.png")
    output_run = tmp_path / "old-run"
    photo(output_run / "copy.png")
    (output_run / ".folder-face-run").touch()
    assert collect_images(Config(tmp_path, tmp_path), threading.Event(), lambda *_: None) == [
        original
    ]


def test_discovery_cancellation(tmp_path):
    photo(tmp_path / "image.png")
    stop = threading.Event()
    stop.set()
    with pytest.raises(Cancelled):
        collect_images(Config(tmp_path, tmp_path), stop, lambda *_: None)


class FakeDeepFace:
    def __init__(self):
        self.represent_calls = 0
        self.extract_calls = 0
        self.verify_calls = 0

    def represent(self, img_path, **kwargs):
        assert kwargs["enforce_detection"] is True
        self.represent_calls += 1
        value = int(img_path[0, 0, 0])
        if value == 0:
            raise ValueError("Face could not be detected")
        if value == 99:
            raise ValueError("Model file is broken")
        embeddings = [[0.0], [1.05]] if value == 30 else [[value / 10]]
        return [
            {"embedding": embedding, "facial_area": {"x": index * 5, "y": 0, "w": 5, "h": 5}}
            for index, embedding in enumerate(embeddings)
        ]

    def extract_faces(self, img_path, **kwargs):
        self.extract_calls += 1
        return [{"facial_area": {"x": 1, "y": 2, "w": 5, "h": 6}}]

    def verify(self, img1_path, img2_path, threshold, **kwargs):
        self.verify_calls += 1
        distance = abs(img1_path[0] - img2_path[0])
        threshold = 0.2 if threshold is None else threshold
        return {"verified": distance <= threshold, "distance": distance, "threshold": threshold}


def test_matcher_caches_references_and_checks_all_faces(tmp_path):
    references = [photo(tmp_path / "ref1.png", 10), photo(tmp_path / "ref2.png", 11)]
    candidate = photo(tmp_path / "candidate.png", 30)
    api = FakeDeepFace()
    matcher = Matcher(Config(tmp_path, tmp_path, references), threading.Event(), api)
    matcher.prepare(lambda *_: None)
    result = matcher.analyze(candidate)
    assert result.status == "candidate_match"
    assert len(result.areas) == 2
    assert result.closest_face == 1
    assert result.supporting_references == 2
    assert api.represent_calls == 3
    assert api.verify_calls == 4


def test_detection_only_never_embeds_or_verifies(tmp_path):
    candidate = photo(tmp_path / "candidate.png")
    api = FakeDeepFace()
    matcher = Matcher(Config(tmp_path, tmp_path), threading.Event(), api)
    matcher.prepare(lambda *_: None)
    result = matcher.analyze(candidate)
    assert result.status == "faces_detected"
    assert len(result.areas) == 1
    assert api.extract_calls == 1
    assert api.represent_calls == api.verify_calls == 0


def test_multiface_reference_rejected(tmp_path):
    reference = photo(tmp_path / "group.png", 30)
    matcher = Matcher(Config(tmp_path, tmp_path, [reference]), threading.Event(), FakeDeepFace())
    with pytest.raises(ValueError, match="Crop"):
        matcher.prepare(lambda *_: None)


def test_no_face_is_distinct_from_model_error(tmp_path):
    matcher = Matcher(Config(tmp_path, tmp_path), threading.Event(), FakeDeepFace())
    with pytest.raises(NoFace):
        matcher.faces(photo(tmp_path / "blank.png", 0), embeddings=True)
    with pytest.raises(ValueError, match="Model file"):
        matcher.faces(photo(tmp_path / "broken.png", 99), embeddings=True)


def test_custom_threshold_prevents_loose_match(tmp_path):
    reference = photo(tmp_path / "ref.png", 10)
    matcher = Matcher(
        Config(tmp_path, tmp_path, [reference], threshold=0.001), threading.Event(), FakeDeepFace()
    )
    matcher.prepare(lambda *_: None)
    assert matcher.analyze(photo(tmp_path / "probe.png", 30)).status == "no_match"


def test_image_orientation_is_normalized_without_changing_file(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 30), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    before = path.read_bytes()
    normalized = Matcher(Config(tmp_path, tmp_path), threading.Event(), FakeDeepFace()).load_image(
        path
    )
    assert normalized.shape == (20, 30, 3)
    assert path.read_bytes() == before


class StubMatcher:
    def __init__(self, config, stop):
        self.config = config
        self.stop = stop

    def prepare(self, emit):
        pass

    def analyze(self, path):
        if path.name == "blank.png":
            raise NoFace()
        if path.name == "corrupt.png":
            raise ValueError("Corrupt image")
        return Result(
            "candidate_match" if self.config.references else "faces_detected",
            [{"x": 0, "y": 0, "w": 10, "h": 10}],
            0.1,
            0.5,
            "ref.png",
            1,
            0,
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_pipeline_reports_errors_copies_matches_and_preserves_originals(tmp_path, workers):
    source = tmp_path / "photos"
    photo(source / "a" / "same.png")
    photo(source / "b" / "same.png")
    photo(source / "blank.png")
    (source / "corrupt.png").write_bytes(b"bad")
    reference = photo(tmp_path / "reference.png")
    originals = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.png")
    }
    events = []
    scan = FolderScan(
        Config(source, source / "reports", [reference], workers=workers),
        lambda name, data: events.append((name, data)),
        threading.Event(),
        StubMatcher,
    )
    scan.run()
    summary = json.loads((scan.run_dir / "summary.json").read_text())
    assert summary["found"] == summary["processed"] == 4
    assert summary["matches"] == 2
    assert summary["errors"] == summary["no_face"] == 1
    assert summary["outcome"] == "complete_with_errors"
    assert (scan.run_dir / "matches" / "a" / "same.png").is_file()
    assert (scan.run_dir / "matches" / "b" / "same.png").is_file()
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in originals.items()
    )
    rows = list(csv.DictReader((scan.run_dir / "results.csv").open()))
    assert {row["status"] for row in rows} == {"candidate_match", "no_face", "processing_error"}
    assert len([data for name, data in events if name == "result"]) == 4


def test_pipeline_detection_only_does_not_copy(tmp_path):
    source = tmp_path / "photos"
    photo(source / "photo.png")
    scan = FolderScan(
        Config(source, tmp_path / "output"), lambda *_: None, threading.Event(), StubMatcher
    )
    scan.run()
    summary = json.loads((scan.run_dir / "summary.json").read_text())
    assert summary["mode"] == "detection"
    assert summary["matches"] == 0
    assert summary["with_faces"] == 1
    assert not (scan.run_dir / "matches").exists()


def test_empty_folder_does_not_load_models(tmp_path):
    def unexpected_matcher(*_):
        raise AssertionError("No model needed for an empty folder")

    source = tmp_path / "photos"
    source.mkdir()
    scan = FolderScan(
        Config(source, tmp_path / "output"), lambda *_: None, threading.Event(), unexpected_matcher
    )
    scan.run()
    assert json.loads((scan.run_dir / "summary.json").read_text())["outcome"] == "complete"
    assert list(csv.DictReader((scan.run_dir / "results.csv").open())) == []


@pytest.mark.parametrize("workers", [1, 2])
def test_cancel_after_first_result_preserves_partial_report(tmp_path, workers):
    source = tmp_path / "photos"
    photo(source / "1.png")
    photo(source / "2.png")
    stop = threading.Event()

    def emit(name, data):
        if name == "result":
            stop.set()

    scan = FolderScan(Config(source, tmp_path / "output", workers=workers), emit, stop, StubMatcher)
    scan.run()
    summary = json.loads((scan.run_dir / "summary.json").read_text())
    assert summary["outcome"] == "stopped"
    assert summary["processed"] == summary["remaining"] == 1
    assert len(list(csv.DictReader((scan.run_dir / "results.csv").open()))) == 1


@pytest.mark.parametrize("workers", [0, -1, 33, 1.5, True])
def test_invalid_workers(tmp_path, workers):
    with pytest.raises(ValueError, match="Parallel workers"):
        Config(tmp_path, tmp_path, workers=workers).validate()


class ParallelMatcher(StubMatcher):
    def analyze(self, path):
        import os
        import time

        # Both workers must enter inference before either is allowed to finish.
        marker = self.config.output / str(os.getpid())
        marker.touch()
        deadline = time.monotonic() + 10
        while len([p for p in self.config.output.iterdir() if p.is_file()]) < 2:
            if time.monotonic() >= deadline:
                raise RuntimeError("Inference did not overlap")
            time.sleep(0.01)
        return Result("faces_detected", [{}], reference=str(os.getpid()))


def test_parallel_inference_uses_separate_processes(tmp_path):
    source = tmp_path / "photos"
    for index in range(4):
        photo(source / f"{index}.png")
    events = []
    scan = FolderScan(
        Config(source, tmp_path / "output", workers=2),
        lambda name, data: events.append((name, data)),
        threading.Event(),
        ParallelMatcher,
    )
    scan.run()
    summary = events[-1][1]
    assert summary["outcome"] == "complete"
    assert summary["processed"] == 4
    results = [data for name, data in events if name == "result"]
    assert len({row["reference"] for row in results}) == 2
    assert len({row["source_path"] for row in results}) == 4
