from __future__ import annotations

import csv
import json
import queue
import threading
import traceback
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .recognition import FaceMatcher, NoFace
from .scraping import Cancelled, CrawlOptions, DownloadedImage, PageScraper


@dataclass(frozen=True)
class ScanConfig:
    url: str
    references: list[Path]
    output: Path
    model: str = "SFace"
    detector: str = "opencv"
    threshold: float | None = None
    selector: str = ""
    browser: bool = False
    delay: float = 0.25
    max_images: int = 0  # Global count of unique image URLs; zero means unlimited.
    page_type: str = "single"
    next_selector: str = ""
    max_pages: int = 100
    max_scrolls: int = 500  # Consecutive attempts without progress, not the lifetime scroll count.
    idle_rounds: int = 3
    wait_seconds: float = 1.5
    max_seconds: float = 1800
    scroll_selector: str = ""
    page_param: str = ""
    page_start: int = 1
    page_step: int = 1
    page_workers: int = 4
    extra_params: str = ""


FIELDS = [
    "status",
    "image_url",
    "page_url",
    "link_url",
    "label",
    "local_path",
    "width",
    "height",
    "distance",
    "threshold",
    "reference",
    "face_count",
    "supporting_references",
    "error",
]


class ScanPipeline:
    """Bounded producer/consumer pipeline, with no GUI dependencies."""

    def __init__(
        self, config: ScanConfig, emit, stop: threading.Event, matcher_factory=FaceMatcher
    ):
        self.config = config
        self.emit = emit
        self.stop = stop
        self.matcher_factory = matcher_factory
        self.pending: queue.Queue[DownloadedImage] = queue.Queue(maxsize=8)
        self.scanner_done = threading.Event()
        self.lock = threading.Lock()
        self.stats = {"found": 0, "downloaded": 0, "processed": 0, "matches": 0, "errors": 0}
        self.run_dir: Path | None = None
        self.log_file = None
        self.discovery_reason = ""
        self.crawl_progress = {"pages": 0, "scrolls": 0, "scroll_attempts": 0}

    def event(self, name: str, payload) -> None:
        # Callbacks can originate in either worker thread. Qt emits them as queued signals.
        with self.lock:
            if name == "crawl_progress":
                self.crawl_progress = {
                    "pages": payload["pages"],
                    "scrolls": payload["scrolls"],
                    "scroll_attempts": payload.get("scroll_attempts", 0),
                }
            if self.log_file:
                self.log_file.write(
                    json.dumps(
                        {"time": datetime.now(UTC).isoformat(), "event": name, "data": payload},
                        default=str,
                    )
                    + "\n"
                )
                self.log_file.flush()
        self.emit(name, payload)

    def bump(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.stats[key] += amount
            snapshot = dict(self.stats)
        self.emit("stats", snapshot)

    @staticmethod
    def source_row(source) -> dict:
        return {
            "image_url": source.url if not source.url.startswith("data:") else "[inline image]",
            "page_url": source.page_url,
            "link_url": source.link_url,
            "label": source.label,
        }

    def scan(self, failures: queue.Queue[dict]) -> None:
        try:
            with PageScraper(self.stop) as scraper:
                options = CrawlOptions(
                    page_type=self.config.page_type,
                    next_selector=self.config.next_selector,
                    max_pages=self.config.max_pages,
                    max_scrolls=self.config.max_scrolls,
                    idle_rounds=self.config.idle_rounds,
                    wait_seconds=self.config.wait_seconds,
                    max_seconds=self.config.max_seconds,
                    scroll_selector=self.config.scroll_selector,
                    page_param=self.config.page_param,
                    page_start=self.config.page_start,
                    page_step=self.config.page_step,
                    page_workers=self.config.page_workers,
                    extra_params=self.config.extra_params,
                )
                with closing(
                    scraper.iter_sources(
                        self.config.url,
                        self.config.browser,
                        self.config.selector,
                        self.event,
                        options,
                    )
                ) as sources:
                    for source in sources:
                        if self.stop.is_set():
                            break
                        self.bump("found")
                        display_url = (
                            source.url if not source.url.startswith("data:") else "Inline image"
                        )
                        self.event("scan_status", f"Downloading: {display_url}")
                        try:
                            image = scraper.download(source, self.run_dir / "images")
                            self.bump("downloaded")
                            self.event("scan_image", {"path": str(image.path), "areas": []})
                            while not self.stop.is_set():
                                try:
                                    self.pending.put(image, timeout=0.2)
                                    break
                                except queue.Full:
                                    self.event(
                                        "scan_status", "Waiting for face processing to catch up…"
                                    )
                        except Cancelled:
                            break
                        except Exception as exc:
                            self.bump("errors")
                            row = {
                                **self.source_row(source),
                                "status": "download_error",
                                "error": str(exc),
                            }
                            failures.put(row)
                            self.event("log", f"Skipped {display_url}: {exc}")
                        if self.stop.is_set():
                            break
                        if self.config.max_images and self.stats["found"] >= self.config.max_images:
                            self.discovery_reason = "image_limit"
                            break
                        if self.stop.wait(self.config.delay):
                            break
                self.discovery_reason = (
                    self.discovery_reason or scraper.finish_reason or "cancelled"
                )
                if not self.stats["found"]:
                    self.event(
                        "log",
                        "No images found. Try browser mode if the page loads images with JavaScript.",
                    )
            self.event(
                "scan_status",
                "Scanning stopped."
                if self.stop.is_set()
                else f"Discovery finished: {self.discovery_reason.replace('_', ' ')}.",
            )
            self.event("log", f"Discovery stop reason: {self.discovery_reason.replace('_', ' ')}.")
        except Cancelled:
            self.discovery_reason = "cancelled"
        except Exception as exc:
            self.discovery_reason = "page_error"
            failures.put({"status": "page_error", "page_url": self.config.url, "error": str(exc)})
            self.bump("errors")
            self.event("error", f"Could not scan the page: {exc}")
            self.event("debug", traceback.format_exc())
        finally:
            self.scanner_done.set()

    def run(self) -> None:
        scanner = None
        failures: queue.Queue[dict] = queue.Queue()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outcome = "complete"
        try:
            self.run_dir = self.config.output / f"{stamp}-{uuid4().hex[:8]}"
            (self.run_dir / "images").mkdir(parents=True)
            (self.run_dir / "config.json").write_text(
                json.dumps(asdict(self.config), default=str, indent=2), encoding="utf-8"
            )
            self.log_file = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
            self.event("run_dir", str(self.run_dir))
            matcher = None
            if self.config.references:
                self.event(
                    "process_status", "Loading DeepFace. First use may download model weights…"
                )
                matcher = self.matcher_factory(
                    self.config.model, self.config.detector, self.config.threshold, self.stop
                )
                matcher.prepare(self.config.references, self.event)
                self.event("process_status", "Reference photos ready. Waiting for images…")
            else:
                self.event("process_status", "Download only. Images will be saved locally.")
                self.event(
                    "log",
                    "No reference photos selected. Downloading images without face recognition.",
                )
            with (self.run_dir / "results.csv").open("w", newline="", encoding="utf-8") as report:
                writer = csv.DictWriter(report, fieldnames=FIELDS)
                writer.writeheader()
                report.flush()
                scanner = threading.Thread(target=self.scan, args=(failures,), name="page-scanner")
                scanner.start()
                while not self.stop.is_set():
                    while not failures.empty():
                        writer.writerow(failures.get_nowait())
                        report.flush()
                    try:
                        image = self.pending.get(timeout=0.2)
                    except queue.Empty:
                        if self.scanner_done.is_set():
                            break
                        continue
                    row = {
                        **self.source_row(image.source),
                        "local_path": str(image.path),
                        "width": image.width,
                        "height": image.height,
                    }
                    self.event(
                        "process_status",
                        ("Detecting and comparing faces: " if matcher else "Saved locally: ")
                        + (image.source.label or image.path.name[:16]),
                    )
                    self.event("process_image", {"path": str(image.path), "areas": []})
                    try:
                        if matcher is None:
                            row["status"] = "downloaded"
                        else:
                            match = matcher.compare(image.path)
                            row.update(asdict(match))
                            row["status"] = "candidate_match" if match.matched else "no_match"
                            self.event(
                                "process_image",
                                {
                                    "path": str(image.path),
                                    "areas": [match.face_area],
                                    "matched": match.matched,
                                },
                            )
                            if match.matched:
                                self.bump("matches")
                    except Cancelled:
                        break
                    except NoFace:
                        row["status"] = "no_face"
                    except Exception as exc:
                        row.update(status="processing_error", error=str(exc))
                        self.bump("errors")
                        self.event("log", f"Face processing failed: {exc}")
                        self.event("debug", traceback.format_exc())
                    writer.writerow({key: row.get(key, "") for key in FIELDS})
                    report.flush()
                    self.bump("processed")
                    self.event("result", row)
                if scanner:
                    # Keep the pipeline alive until network/browser cleanup has finished.
                    scanner.join()
                while not failures.empty():
                    writer.writerow(failures.get_nowait())
                report.flush()
                if self.stats["errors"]:
                    outcome = "complete_with_errors"
        except Cancelled:
            outcome = "stopped"
        except Exception as exc:
            outcome = "failed"
            self.bump("errors")
            self.event("error", str(exc))
            self.event("debug", traceback.format_exc())
            self.stop.set()
        finally:
            if scanner and scanner.is_alive():
                self.stop.set()
                scanner.join()
            if self.stop.is_set() and outcome not in {"failed"}:
                outcome = "stopped"
            summary = {
                "outcome": outcome,
                "mode": "face_matching" if self.config.references else "download_only",
                "page_type": self.config.page_type,
                "discovery_reason": self.discovery_reason,
                **self.crawl_progress,
                **self.stats,
                "pending": self.pending.qsize(),
                "unprocessed": max(0, self.stats["downloaded"] - self.stats["processed"]),
                "run_dir": str(self.run_dir) if self.run_dir else "",
            }
            try:
                if self.run_dir and self.run_dir.exists():
                    (self.run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
            except OSError as exc:
                self.emit("error", f"Could not save summary: {exc}")
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.emit("done", summary)
