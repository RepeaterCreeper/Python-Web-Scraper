from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
Image.MAX_IMAGE_PIXELS = 25_000_000
FIELDS = [
    "status",
    "source_path",
    "relative_path",
    "face_count",
    "distance",
    "threshold",
    "reference",
    "supporting_references",
    "copied_path",
    "error",
]


class Cancelled(Exception):
    pass


class NoFace(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    source: Path
    output: Path
    references: list[Path] = field(default_factory=list)
    recursive: bool = True
    model: str = "SFace"
    detector: str = "opencv"
    threshold: float | None = None
    copy_matches: bool = True
    workers: int = 1

    def validate(self):
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("Parallel workers must be an integer between 1 and 32.")
        if not self.source.is_dir():
            raise ValueError("Choose an existing folder of images.")
        if any(not path.is_file() for path in self.references):
            raise ValueError("A reference photo no longer exists. Choose it again.")
        if self.output.exists() and not self.output.is_dir():
            raise ValueError("The output path must be a folder.")


@dataclass(frozen=True)
class Result:
    status: str
    areas: list[dict]
    distance: float | None = None
    threshold: float | None = None
    reference: str = ""
    supporting_references: int = 0
    closest_face: int | None = None


def collect_images(config: Config, stop: threading.Event, emit) -> list[Path]:
    source, output = config.source.resolve(), config.output.resolve()
    references = {path.resolve() for path in config.references}
    # An output folder inside the source must not become new input on later scans.
    exclude_output = output != source and output.is_relative_to(source)
    images = []

    def walk_error(error):
        raise OSError(f"Cannot read image folder: {error}") from error

    for directory, directories, files in os.walk(source, followlinks=False, onerror=walk_error):
        if stop.is_set():
            raise Cancelled()
        current = Path(directory)
        directories[:] = sorted(
            name
            for name in directories
            if not (current / name).is_symlink()
            and not (current / name / ".folder-face-run").is_file()
            and not (exclude_output and (current / name).resolve() == output)
        )
        for name in sorted(files):
            if stop.is_set():
                raise Cancelled()
            path = current / name
            if (
                path.suffix.lower() in EXTENSIONS
                and not path.is_symlink()
                and path.resolve() not in references
            ):
                images.append(path)
        emit("discovery", len(images))
        if not config.recursive:
            break
    return images


class Matcher:
    """DeepFace adapter; reference embeddings live only in memory for this scan."""

    def __init__(self, config: Config, stop: threading.Event, api=None):
        if api is None:
            from deepface import DeepFace

            api = DeepFace
        self.api, self.config, self.stop = api, config, stop
        self.references: list[tuple[str, list[float]]] = []

    def load_image(self, path: Path):
        import numpy as np

        if self.stop.is_set():
            raise Cancelled()
        with Image.open(path) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 25-megapixel limit.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])

    def faces(self, path: Path, embeddings: bool) -> list[dict]:
        image = self.load_image(path)
        try:
            options = dict(
                img_path=image,
                detector_backend=self.config.detector,
                enforce_detection=True,
                align=True,
            )
            if embeddings:
                faces = self.api.represent(**options, model_name=self.config.model)
            else:
                faces = self.api.extract_faces(**options)
        except ValueError as exc:
            if "face could not be detected" in str(exc).lower() or "no face" in str(exc).lower():
                raise NoFace("No face detected.") from exc
            raise
        if not faces:
            raise NoFace("No face detected.")
        return faces

    def prepare(self, emit):
        if self.config.workers > 1 and hasattr(self.api, "build_model"):
            # Finish any first-use weight downloads before starting child processes.
            emit("status", "Preparing models for parallel workers…")
            self.api.build_model(self.config.detector, task="face_detector")
            if self.config.references:
                self.api.build_model(self.config.model)
        for path in self.config.references:
            emit("status", f"Preparing reference: {path.name}")
            emit("preview", {"path": str(path), "areas": []})
            faces = self.faces(path, embeddings=True)
            if len(faces) != 1:
                raise ValueError(
                    f"Reference {path.name} contains {len(faces)} faces. Crop it to one face."
                )
            self.references.append((str(path.resolve()), [float(x) for x in faces[0]["embedding"]]))

    def analyze(self, path: Path) -> Result:
        faces = self.faces(path, embeddings=bool(self.references))
        areas = [face["facial_area"] for face in faces]
        if not self.references:
            return Result("faces_detected", areas)
        comparisons = []
        support: dict[int, int] = {}
        for index, face in enumerate(faces):
            vector = [float(x) for x in face["embedding"]]
            for reference, reference_vector in self.references:
                if self.stop.is_set():
                    raise Cancelled()
                match = self.api.verify(
                    img1_path=reference_vector,
                    img2_path=vector,
                    model_name=self.config.model,
                    distance_metric="cosine",
                    threshold=self.config.threshold,
                    silent=True,
                )
                support[index] = support.get(index, 0) + int(match["verified"])
                comparisons.append((match, reference, index))
        best, reference, index = min(comparisons, key=lambda item: item[0]["distance"])
        return Result(
            "candidate_match" if best["verified"] else "no_match",
            areas,
            max(0.0, float(best["distance"])),
            float(best["threshold"]),
            reference,
            support.get(index, 0),
            index,
        )


_worker_matcher = None


def _initialize_worker(config, stop, matcher_factory, references):
    global _worker_matcher
    _worker_matcher = matcher_factory(config, stop)
    _worker_matcher.references = references


def _analyze_worker(path):
    return _worker_matcher.analyze(path)


def analyze_images(config, images, matcher, matcher_factory, stop):
    """Yield completed futures; only the coordinator emits events or writes reports."""
    if not images:
        return
    if config.workers == 1:
        for path in images:
            if stop.is_set():
                raise Cancelled()
            future = Future()
            try:
                future.set_result(matcher.analyze(path))
            except Exception as exc:
                future.set_exception(exc)
            yield path, future
        return

    # Spawn avoids inheriting Qt/native model threads from the desktop process.
    context = multiprocessing.get_context("spawn")
    worker_stop = context.Event()
    pool = ProcessPoolExecutor(
        max_workers=min(config.workers, len(images)),
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(config, worker_stop, matcher_factory, getattr(matcher, "references", [])),
    )
    pending = {}
    paths = iter(images)

    def submit_next():
        if not stop.is_set():
            path = next(paths, None)
            if path is not None:
                pending[pool.submit(_analyze_worker, path)] = path

    try:
        for _ in range(min(config.workers, len(images))):
            submit_next()
        while pending:
            if stop.is_set():
                raise Cancelled()
            done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                if stop.is_set():
                    raise Cancelled()
                path = pending.pop(future)
                yield path, future
                submit_next()
    finally:
        worker_stop.set()
        pool.shutdown(wait=True, cancel_futures=True)


class FolderScan:
    def __init__(self, config: Config, emit, stop: threading.Event, matcher_factory=Matcher):
        self.config, self.emit, self.stop = config, emit, stop
        self.matcher_factory = matcher_factory
        self.run_dir: Path | None = None
        self.log_file = None
        self.stats = {
            "found": 0,
            "processed": 0,
            "matches": 0,
            "with_faces": 0,
            "no_face": 0,
            "errors": 0,
        }

    def event(self, name, data):
        if self.log_file:
            self.log_file.write(
                json.dumps(
                    {"time": datetime.now(UTC).isoformat(), "event": name, "data": data},
                    default=str,
                )
                + "\n"
            )
            self.log_file.flush()
        self.emit(name, data)

    def run(self):
        outcome = "complete"
        try:
            self.config.validate()
            self.run_dir = self.config.output.resolve() / (
                datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
            )
            self.run_dir.mkdir(parents=True)
            (self.run_dir / ".folder-face-run").touch()
            (self.run_dir / "config.json").write_text(
                json.dumps(asdict(self.config), default=str, indent=2), encoding="utf-8"
            )
            self.log_file = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
            self.event("run_dir", str(self.run_dir))
            self.event("status", "Finding image files…")
            images = collect_images(self.config, self.stop, self.event)
            self.stats["found"] = len(images)
            self.event("stats", dict(self.stats))
            with (self.run_dir / "results.csv").open("w", newline="", encoding="utf-8") as report:
                writer = csv.DictWriter(report, fieldnames=FIELDS)
                writer.writeheader()
                report.flush()
                matcher = None
                if images:
                    self.event("status", "Loading DeepFace. First use may download model weights…")
                    matcher = self.matcher_factory(self.config, self.stop)
                    matcher.prepare(self.event)
                    self.event(
                        "status",
                        f"Recognizing images with {min(self.config.workers, len(images))} worker(s)…",
                    )
                else:
                    self.event(
                        "log",
                        "No supported candidate images found. Reference photos, symbolic links and scan-output folders are excluded.",
                    )
                with closing(
                    analyze_images(self.config, images, matcher, self.matcher_factory, self.stop)
                ) as completed:
                    for path, future in completed:
                        if self.stop.is_set():
                            raise Cancelled()
                        relative = path.relative_to(self.config.source.resolve())
                        row = {"source_path": str(path), "relative_path": str(relative)}
                        self.event(
                            "status",
                            f"Processing {self.stats['processed'] + 1}/{len(images)}: {relative}",
                        )
                        self.event("preview", {"path": str(path), "areas": []})
                        try:
                            result = future.result()
                            row.update(
                                status=result.status,
                                face_count=len(result.areas),
                                distance=result.distance,
                                threshold=result.threshold,
                                reference=result.reference,
                                supporting_references=result.supporting_references,
                            )
                            self.stats["with_faces"] += 1
                            self.event(
                                "preview",
                                {
                                    "path": str(path),
                                    "areas": result.areas,
                                    "closest_face": result.closest_face,
                                    "matched": result.status == "candidate_match",
                                },
                            )
                            if result.status == "candidate_match":
                                self.stats["matches"] += 1
                                if self.config.copy_matches:
                                    destination = self.run_dir / "matches" / relative
                                    try:
                                        destination.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(path, destination)
                                        row["copied_path"] = str(destination)
                                    except OSError as exc:
                                        self.stats["errors"] += 1
                                        row["error"] = f"Matched, but could not copy: {exc}"
                                        self.event("log", row["error"])
                        except NoFace:
                            row.update(status="no_face", face_count=0)
                            self.stats["no_face"] += 1
                        except Cancelled:
                            raise
                        except Exception as exc:
                            row.update(status="processing_error", error=str(exc))
                            self.stats["errors"] += 1
                            self.event("log", f"Could not process {relative}: {exc}")
                            self.event("debug", traceback.format_exc())
                        writer.writerow({key: row.get(key, "") for key in FIELDS})
                        report.flush()
                        self.stats["processed"] += 1
                        self.event("stats", dict(self.stats))
                        self.event("result", row)
            if self.stats["errors"]:
                outcome = "complete_with_errors"
        except Cancelled:
            outcome = "stopped"
        except Exception as exc:
            outcome = "failed"
            self.stats["errors"] += 1
            self.event("error", str(exc))
            self.event("debug", traceback.format_exc())
        finally:
            summary = {
                "outcome": outcome,
                "mode": "matching" if self.config.references else "detection",
                **self.stats,
                "remaining": max(0, self.stats["found"] - self.stats["processed"]),
                "run_dir": str(self.run_dir or ""),
            }
            try:
                if self.run_dir:
                    (self.run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
            except OSError as exc:
                self.emit("error", f"Could not save summary: {exc}")
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.emit("done", summary)
