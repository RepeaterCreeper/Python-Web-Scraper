# Folder Face Finder

A separate, standalone Python desktop app for scanning local image folders with
DeepFace. It does not depend on or import the SScraper application.

## Run in this workspace

From the SScraper project folder, reuse the already-installed Python environment:

```sh
.venv/bin/python folder_face_finder/run.py
```

To install it separately, copy this entire folder wherever you want and use
Python 3.11 or 3.12:

```sh
cd folder_face_finder
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run.py
```

Or use `uv sync` followed by `uv run folder-face-finder` from this folder.

## Use

1. Choose the **Image folder**. **Include subfolders** is enabled by default.
2. For identity matching, add clear reference photos of the same person, with
   exactly one face in each. Multiple references help cover different angles.
   With no references, the app performs **face detection only**, counting and
   highlighting faces without identifying anyone.
3. Choose the output folder. The default is this app's `runs/` folder.
4. Click **Start scan**. The current image, face boxes, progress and results
   update while DeepFace runs in a background thread.
5. Filter results to candidates, images with faces, no-face images or errors.
   Double-click a row to open the original image. **Open results folder** opens
   the reports and any copied candidate matches.

Original images are read and never moved, renamed, or edited. **Copy candidate
matches into the results folder** is optional and preserves relative subfolders,
so matching files with the same name do not collide. Candidate matches require
manual review; they are not confirmed identities or calibrated probabilities.

**Stop** preserves completed results and stops before the next image or
comparison. A currently running model import, model download, inference, or file
copy must finish first. Closing the app during a scan requests the same graceful stop.

## How matching works

The default model is **SFace** within **DeepFace**, with the OpenCV detector.
ArcFace, Facenet512 and VGG-Face are also selectable. References are embedded once
per scan using `DeepFace.represent()`. Every detected candidate face is compared
against each reference with `DeepFace.verify()` and cosine distance. The closest
comparison supplies the result; a candidate matches if any reference passes the
model's threshold. The Refs column counts supporting references for that face.
An optional custom distance threshold overrides the model's default; lower
thresholds require closer similarity. Detection-only uses `DeepFace.extract_faces()`.

Photos are normalized in memory for EXIF orientation and RGB/BGR conventions.
Supported extensions: JPG/JPEG, PNG, WebP, BMP, TIFF and GIF. For animated images,
only the first frame is analyzed. Images over 25 megapixels and unreadable files
are recorded as errors; no detected face gets its own `no_face` status.

Reference files are excluded from candidate scans when they are inside the input
folder, preventing automatic self-matches. Symbolic links are excluded. Output
folders inside the input and marked scan-result subfolders are also excluded, so
copied matches are not rescanned. Unreadable directories fail discovery rather
than silently claiming a full scan.

## Reports and privacy

Each scan gets a new output folder:

```text
runs/<timestamp>-<id>/
  config.json        # Local paths, model, detector, options
  results.csv        # Status, original path, face count, distance, threshold, errors
  events.jsonl       # Timestamped scan activity and diagnostics
  summary.json       # Outcome, counts, remaining images
  matches/           # Optional copies of candidate matches
```

CSV rows are flushed after each image. Reference embeddings stay in memory.
The app does not upload photos to an API. DeepFace may download model weights on
first use and caches them under `~/.deepface/weights`.

## Checks

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check face_finder tests run.py
```

Automated tests use stub inference to run without model downloads. They cover
recursive discovery and exclusions, multiple faces and references, no-face/error
handling, reports, non-destructive match copying, cancellation and detection-only mode.
