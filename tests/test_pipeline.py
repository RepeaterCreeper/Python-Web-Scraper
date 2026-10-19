import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from sscraper.pipeline import ScanConfig, ScanPipeline
from sscraper.recognition import FaceMatch, NoFace


@pytest.fixture
def website():
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, "PNG")
    image = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/gallery":
                content = b'<img src="/portrait.png"><img src="/blank.png"><img src="/bad.png"><a href="/other">Another page</a>'
                kind = "text/html"
            elif self.path == "/page/1":
                content = b'<img src="/portrait.png"><a rel="next" href="/page/2">Next</a>'
                kind = "text/html"
            elif self.path == "/page/2":
                content = b'<img src="/portrait.png"><img src="/blank.png"><a rel="next" href="/page/1">Next</a>'
                kind = "text/html"
            elif self.path in {"/portrait.png", "/blank.png"}:
                content, kind = image, "image/png"
            elif self.path == "/bad.png":
                content, kind = b"not an image", "text/plain"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/gallery"
    server.shutdown()
    server.server_close()
    thread.join()


class StubMatcher:
    def __init__(self, *_):
        self.count = 0

    def prepare(self, *_):
        pass

    def compare(self, path):
        self.count += 1
        if self.count == 2:
            raise NoFace()
        return FaceMatch(True, 0.1, 0.5, "reference.jpg", 1, {"x": 0, "y": 0, "w": 10, "h": 10}, 1)


def test_single_page_pipeline_saves_results_and_failure(website, tmp_path):
    events = []
    pipeline = ScanPipeline(
        ScanConfig(website, [Path("reference.jpg")], tmp_path, delay=0),
        lambda name, data: events.append((name, data)),
        threading.Event(),
        StubMatcher,
    )
    pipeline.run()
    done = next(data for name, data in events if name == "done")
    assert done["found"] == 3
    assert done["downloaded"] == done["processed"] == 2
    assert done["matches"] == 1
    assert done["errors"] == 1
    assert done["outcome"] == "complete_with_errors"
    rows = list(csv.DictReader((pipeline.run_dir / "results.csv").open()))
    assert {row["status"] for row in rows} == {"candidate_match", "no_face", "download_error"}
    assert json.loads((pipeline.run_dir / "summary.json").read_text())["pending"] == 0
    assert any(name == "scan_image" for name, _ in events)
    assert any(name == "process_image" for name, _ in events)


def test_cancel_under_backpressure_does_not_deadlock(monkeypatch, tmp_path):
    from sscraper.scraping import DownloadedImage, ImageSource, PageScraper

    stop = threading.Event()
    # More images than the queue holds, with a deliberately slow consumer.
    sources = [ImageSource(f"https://example.com/{i}.jpg", "", "", "") for i in range(50)]
    monkeypatch.setattr(PageScraper, "iter_sources", lambda *_: (source for source in sources))
    monkeypatch.setattr(
        PageScraper,
        "download",
        lambda _, source, directory: DownloadedImage(source, directory / "sample.jpg", 100, 80),
    )

    class SlowMatcher(StubMatcher):
        def compare(self, path):
            stop.wait(1)
            return super().compare(path)

    pipeline = ScanPipeline(
        ScanConfig("https://example.com", [Path("reference.jpg")], tmp_path, delay=0),
        lambda *_: None,
        stop,
        SlowMatcher,
    )
    thread = threading.Thread(target=pipeline.run)
    thread.start()
    assert pipeline.scanner_done.wait(0.1) is False
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert json.loads((pipeline.run_dir / "summary.json").read_text())["outcome"] == "stopped"
    summary = json.loads((pipeline.run_dir / "summary.json").read_text())
    assert summary["unprocessed"] == summary["downloaded"] - summary["processed"]


def test_no_references_downloads_images_without_loading_deepface(website, tmp_path):
    def unexpected_matcher(*_):
        raise AssertionError("Download-only mode must not load DeepFace.")

    events = []
    pipeline = ScanPipeline(
        ScanConfig(website, [], tmp_path, delay=0),
        lambda name, data: events.append((name, data)),
        threading.Event(),
        unexpected_matcher,
    )
    pipeline.run()
    rows = list(csv.DictReader((pipeline.run_dir / "results.csv").open()))
    downloaded = [row for row in rows if row["status"] == "downloaded"]
    assert len(downloaded) == 2
    assert all(Path(row["local_path"]).is_file() for row in downloaded)
    assert all(row["distance"] == row["reference"] == "" for row in downloaded)
    summary = json.loads((pipeline.run_dir / "summary.json").read_text())
    assert summary["mode"] == "download_only"
    assert summary["downloaded"] == summary["processed"] == 2
    assert summary["matches"] == summary["unprocessed"] == 0
    assert not any("Loading DeepFace" in str(data) for _, data in events)


@pytest.mark.parametrize(
    "image_limit,reason,total,pages", [(0, "repeated_page", 2, 2), (1, "image_limit", 1, 1)]
)
def test_pagination_pipeline_saves_images_and_obeys_global_limit(
    website, tmp_path, image_limit, reason, total, pages
):
    base = website.removesuffix("/gallery")
    pipeline = ScanPipeline(
        ScanConfig(
            base + "/page/1",
            [],
            tmp_path,
            delay=0,
            page_type="pagination",
            wait_seconds=0.01,
            max_images=image_limit,
        ),
        lambda *_: None,
        threading.Event(),
    )
    pipeline.run()
    summary = json.loads((pipeline.run_dir / "summary.json").read_text())
    assert summary["discovery_reason"] == reason
    assert summary["pages"] == pages
    assert summary["found"] == summary["downloaded"] == summary["processed"] == total
    rows = list(csv.DictReader((pipeline.run_dir / "results.csv").open()))
    assert len(rows) == total
    assert all(row["status"] == "downloaded" for row in rows)
    if not image_limit:
        assert rows[1]["page_url"] == base + "/page/2"
