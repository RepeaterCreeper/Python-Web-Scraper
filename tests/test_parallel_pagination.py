import csv
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image

from sscraper.pipeline import ScanConfig, ScanPipeline
from sscraper.scraping import Cancelled, CrawlOptions, PageScraper, parameter_page_url


@pytest.fixture
def parameter_site():
    state = {"active": 0, "peak": 0, "queries": []}
    lock = threading.Lock()
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="PNG")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path.endswith(".png"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(image.getvalue())
                return
            query = parse_qs(parsed.query)
            with lock:
                state["queries"].append(query)
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.12)
                number = int(query.get("page", query.get("offset", ["1"]))[0])
                if parsed.path == "/repeat":
                    number = 1
                if parsed.path == "/error" and number == 2:
                    self.send_error(500)
                    return
                content = (
                    f'<img src="/shared.png"><img src="/{number}.png">'
                    if number <= 10
                    else "<p>End</p>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(content.encode())
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def test_query_merge_preserves_filters_and_encodes_values():
    options = CrawlOptions(
        page_param="offset",
        page_start=0,
        page_step=20,
        extra_params="?sort=newest&tag=a%26b&tag=c&blank=&offset=999",
    )
    url = parameter_page_url(
        "https://example.com/list?sort=old&keep=x&keep=y&offset=5#section", options, 2
    )
    assert parse_qs(urlsplit(url).query, keep_blank_values=True) == {
        "sort": ["newest"],
        "keep": ["x", "y"],
        "tag": ["a&b", "c"],
        "blank": [""],
        "offset": ["40"],
    }
    assert not urlsplit(url).fragment


def test_real_parallel_pages_are_bounded_ordered_and_deduplicated(parameter_site):
    base, state = parameter_site
    progress = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/list?filter=cats",
                False,
                "",
                lambda n, d: progress.append((n, d)),
                CrawlOptions(
                    page_type="pagination",
                    page_param="page",
                    max_pages=5,
                    page_workers=3,
                    wait_seconds=0.01,
                    extra_params="sort=newest",
                ),
            )
        )
        assert scraper.finish_reason == "page_limit"
    assert 2 <= state["peak"] <= 3
    assert len(state["queries"]) == 5
    assert all(q["filter"] == ["cats"] and q["sort"] == ["newest"] for q in state["queries"])
    assert [i.url for i in images] == [base + "/shared.png"] + [
        f"{base}/{n}.png" for n in range(1, 6)
    ]
    assert [d["pages"] for n, d in progress if n == "crawl_progress"] == list(range(1, 6))
    assert images[-1].page_url.endswith("filter=cats&sort=newest&page=5")


@pytest.mark.parametrize(
    "path,start,reason,count",
    [
        ("/repeat", 1, "repeated_page", 2),
        ("/list", 10, "no_new_images", 2),
    ],
)
def test_parameter_end_conditions(parameter_site, path, start, reason, count):
    base, state = parameter_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + path,
                False,
                "",
                lambda *_: None,
                CrawlOptions(
                    page_type="pagination",
                    page_param="page",
                    page_start=start,
                    page_workers=3,
                    max_pages=20,
                    wait_seconds=0.01,
                ),
            )
        )
        assert scraper.finish_reason == reason
    assert len(images) == count
    assert len(state["queries"]) <= 3


@pytest.mark.parametrize("cancel", [False, True])
def test_early_close_and_cancel_stop_after_current_batch(parameter_site, cancel):
    base, state = parameter_site
    stop = threading.Event()
    with PageScraper(stop) as scraper:
        sources = scraper.iter_sources(
            base,
            False,
            "",
            lambda *_: None,
            CrawlOptions(
                page_type="pagination",
                page_param="page",
                max_pages=100,
                page_workers=2,
                wait_seconds=0.01,
            ),
        )
        next(sources)
        if cancel:
            stop.set()
            with pytest.raises(Cancelled):
                next(sources)
        sources.close()
    assert len(state["queries"]) <= 2
    assert state["active"] == 0
    assert not any(t.name.startswith("pages_") for t in threading.enumerate())


def test_parameter_time_limit(parameter_site):
    base, state = parameter_site
    with PageScraper(threading.Event()) as scraper:
        assert (
            list(
                scraper.iter_sources(
                    base,
                    False,
                    "",
                    lambda *_: None,
                    CrawlOptions(page_type="pagination", page_param="page", max_seconds=0.05),
                )
            )
            == []
        )
        assert scraper.finish_reason == "time_limit"
    assert len(state["queries"]) <= 4


@pytest.mark.parametrize(
    "image_limit,path,reason",
    [
        (1, "/list", "image_limit"),
        (0, "/error", "page_error"),
    ],
)
def test_parallel_pipeline_preserves_reports_and_limits(
    parameter_site, tmp_path, image_limit, path, reason
):
    base, state = parameter_site
    config = ScanConfig(
        base + path,
        [],
        tmp_path,
        page_type="pagination",
        page_param="page",
        page_workers=2,
        max_pages=10,
        max_images=image_limit,
        delay=0,
        wait_seconds=0.01,
    )
    pipeline = ScanPipeline(config, lambda *_: None, threading.Event())
    pipeline.run()
    summary = json.loads((pipeline.run_dir / "summary.json").read_text())
    assert summary["discovery_reason"] == reason
    assert summary["downloaded"] == summary["processed"] == (1 if image_limit else 2)
    with (pipeline.run_dir / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "downloaded"
    assert len(state["queries"]) <= 2


def test_parallel_browser_pages(parameter_site):
    from pathlib import Path

    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed")
    base, _ = parameter_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base,
                True,
                "",
                lambda *_: None,
                CrawlOptions(
                    page_type="pagination",
                    page_param="page",
                    max_pages=2,
                    page_workers=2,
                    idle_rounds=1,
                    wait_seconds=0.1,
                ),
            )
        )
        assert scraper.finish_reason == "page_limit"
    assert [i.url for i in images] == [base + s for s in ("/shared.png", "/1.png", "/2.png")]
