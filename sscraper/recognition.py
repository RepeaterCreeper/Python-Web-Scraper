from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .scraping import Cancelled


class NoFace(ValueError):
    pass


@dataclass(frozen=True)
class FaceMatch:
    matched: bool
    distance: float
    threshold: float
    reference: str
    face_count: int
    face_area: dict
    supporting_references: int


class FaceMatcher:
    """Compute reference embeddings once; detect every candidate face once.

    DeepFace.verify compares cached embeddings and supplies its calibrated threshold.
    Reference images must contain exactly one face to avoid silently selecting another person.
    """

    def __init__(
        self,
        model: str,
        detector: str,
        threshold: float | None,
        stop: threading.Event,
        deepface=None,
    ):
        if deepface is None:
            if sys.version_info >= (3, 14):
                raise RuntimeError(
                    "Face matching requires Python 3.11-3.13 because TensorFlow does not "
                    "yet support this Python version. Leave reference photos empty to "
                    "download images, or run the app with Python 3.13 for face matching."
                )
            from deepface import DeepFace

            deepface = DeepFace
        self.deepface = deepface
        self.model = model
        self.detector = detector
        self.threshold = threshold
        self.stop = stop
        self.references: list[tuple[str, list[float]]] = []

    def represent(self, path: Path) -> list[dict]:
        if self.stop.is_set():
            raise Cancelled()
        try:
            faces = self.deepface.represent(
                img_path=str(path),
                model_name=self.model,
                detector_backend=self.detector,
                enforce_detection=True,
                align=True,
            )
        except ValueError as exc:
            if "face could not be detected" in str(exc).lower() or "no face" in str(exc).lower():
                raise NoFace(f"No face detected in {path.name}.") from exc
            raise
        if not faces:
            raise NoFace(f"No face detected in {path.name}.")
        return faces

    def prepare(self, references: list[Path], emit) -> None:
        for path in references:
            emit("process_status", f"Preparing reference: {path.name}")
            emit("process_image", {"path": str(path), "areas": []})
            faces = self.represent(path)
            if len(faces) != 1:
                raise ValueError(
                    f"Reference {path.name} contains {len(faces)} faces. Crop it to just the target face."
                )
            self.references.append((str(path), [float(x) for x in faces[0]["embedding"]]))
        if not self.references:
            raise ValueError("Choose at least one reference photo.")

    def compare(self, path: Path) -> FaceMatch:
        faces = self.represent(path)
        comparisons = []
        support_per_face: dict[int, set[str]] = {}
        for index, face in enumerate(faces):
            embedding = [float(x) for x in face["embedding"]]
            for reference, reference_embedding in self.references:
                if self.stop.is_set():
                    raise Cancelled()
                result = self.deepface.verify(
                    img1_path=reference_embedding,
                    img2_path=embedding,
                    model_name=self.model,
                    distance_metric="cosine",
                    threshold=self.threshold,
                    silent=True,
                )
                if result["verified"]:
                    support_per_face.setdefault(index, set()).add(reference)
                comparisons.append((result, reference, face, index))
        # All comparisons use the same model, metric and threshold; the lowest distance wins.
        best, reference, face, index = min(comparisons, key=lambda item: item[0]["distance"])
        return FaceMatch(
            bool(best["verified"]),
            float(best["distance"]),
            float(best["threshold"]),
            reference,
            len(faces),
            face["facial_area"],
            len(support_per_face.get(index, set())),
        )
