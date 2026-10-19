import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sscraper.scraping import Cancelled, CrawlOptions, PageScraper, next_page_url


@pytest.fixture
def crawl_site():
    requests_seen = []
    pages = {
        "/pages/1": '<img src="/shared.png"><img src="/1.png"><a rel="next" href="2">Next</a>',
        "/pages/2": '<img src="/shared.png"><img src="/2.png"><a rel="next" href="3">Next</a>',
        "/pages/3": '<img src="/shared.png"><img src="/3.png">',
        "/cycle/1": '<img src="/a.png"><a rel="next" href="2">Next</a>',
        "/cycle/2": '<img src="/b.png"><a rel="next" href="1">Next</a>',
        "/outside": '<img src="/a.png"><a rel="next" href="https://different.example/page">Next</a>',
        "/custom": '<img src="/a.png"><a class="more" href="/pages/3">More</a>',
        "/buttons": """<div id="listing"><img src="/0.png"></div>
            <button id="next" onclick="step()">Next</button><script>
            let n = 0; function step() { n++;
                document.querySelector('#listing').innerHTML = `<img src="/${n}.png">`;
                if (n === 2) document.querySelector('#next').disabled = true;
            }</script>""",
        "/repeating-button": """<img src="/0.png"><button id="next" onclick="void(0)">Next</button>""",
        "/null-scroller": """<img src="/0.png"><script>
            Object.defineProperty(document, 'scrollingElement', {get: () => null});
            </script>""",
        "/tall-feed": """<div id="feed" style="height:180px;overflow:auto">
            <div style="height:2500px"><img src="/0.png"></div></div>""",
        "/infinite": """<style>#feed{height:180px;overflow:auto}.entry{height:220px}</style>
            <div id="feed"><div class="entry"><img src="/0.png"></div></div>
            <script>const feed = document.querySelector('#feed'); let n=0, loading=false;
            feed.addEventListener('scroll', () => {
                if (loading || n>=3 || feed.scrollTop+feed.clientHeight<feed.scrollHeight-5) return;
                loading=true;
                setTimeout(() => { n++;
                    feed.insertAdjacentHTML('beforeend', `<div class="entry"><img src="/${n}.png"></div>`);
                    loading=false;
                }, 30);
            });</script>""",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests_seen.append(self.path)
            if self.path in pages:
                content = pages[self.path].encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests_seen
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def chromium_available():
    from pathlib import Path

    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed; run python -m playwright install chromium")


def test_next_link_detection_and_custom_selector():
    assert (
        next_page_url('<a rel="next" href="?page=2">Continue</a>', "https://example.com/gallery")
        == "https://example.com/gallery?page=2"
    )
    assert (
        next_page_url('<a class="more" href="/2">More</a>', "https://example.com/1", ".more")
        == "https://example.com/2"
    )
    assert (
        next_page_url(
            '<a aria-disabled="true" rel="next" href="/2">Next</a>', "https://example.com"
        )
        is None
    )
    with pytest.raises(ValueError, match="browser"):
        next_page_url('<button id="next">Next</button>', "https://example.com", "#next")


def test_pagination_collects_unique_images_and_source_pages(crawl_site):
    base, requests_seen = crawl_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/pages/1",
                False,
                "",
                lambda *_: None,
                CrawlOptions(page_type="pagination", wait_seconds=0.01),
            )
        )
        assert scraper.finish_reason == "no_next_page"
    assert [image.url for image in images] == [
        base + suffix for suffix in ("/shared.png", "/1.png", "/2.png", "/3.png")
    ]
    assert images[-1].page_url == base + "/pages/3"
    assert requests_seen == ["/pages/1", "/pages/2", "/pages/3"]


@pytest.mark.parametrize(
    "path,limit,reason,count",
    [
        ("/cycle/1", 100, "repeated_page", 2),
        ("/pages/1", 1, "page_limit", 2),
        ("/outside", 100, "next_page_outside_site", 1),
    ],
)
def test_pagination_stops_at_cycle_limit_or_outside_site(crawl_site, path, limit, reason, count):
    base, _ = crawl_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + path,
                False,
                "",
                lambda *_: None,
                CrawlOptions(page_type="pagination", max_pages=limit, wait_seconds=0.01),
            )
        )
        assert scraper.finish_reason == reason
    assert len(images) == count


def test_discovery_time_limit_stops_before_fetch(crawl_site):
    base, requests_seen = crawl_site
    with PageScraper(threading.Event()) as scraper:
        assert (
            list(
                scraper.iter_sources(
                    base + "/pages/1", False, "", lambda *_: None, CrawlOptions(max_seconds=1e-12)
                )
            )
            == []
        )
        assert scraper.finish_reason == "time_limit"
    assert requests_seen == []


def test_cancellation_between_pages(crawl_site):
    base, requests_seen = crawl_site
    stop = threading.Event()
    with PageScraper(stop) as scraper:
        images = scraper.iter_sources(
            base + "/pages/1",
            False,
            "",
            lambda *_: None,
            CrawlOptions(page_type="pagination", wait_seconds=0.01),
        )
        assert next(images).url == base + "/shared.png"
        stop.set()
        with pytest.raises(Cancelled):
            next(images)
    assert requests_seen == ["/pages/1"]


def test_real_browser_infinite_scroll_collects_each_batch(crawl_site, chromium_available):
    base, _ = crawl_site
    progress = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/infinite",
                True,
                "#feed",
                lambda name, data: progress.append((name, data)),
                CrawlOptions(
                    page_type="infinite", scroll_selector="#feed", idle_rounds=2, wait_seconds=0.1
                ),
            )
        )
        assert scraper.finish_reason == "no_new_images"
    assert [image.url for image in images] == [f"{base}/{index}.png" for index in range(4)]
    assert sum(name == "crawl_progress" and data["new_images"] > 0 for name, data in progress) == 4


def test_real_browser_scroll_limit(crawl_site, chromium_available):
    base, _ = crawl_site
    progress = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/infinite",
                True,
                "#feed",
                lambda name, data: progress.append((name, data)),
                CrawlOptions(
                    page_type="infinite",
                    scroll_selector="#feed",
                    max_scrolls=2,
                    idle_rounds=10,
                    wait_seconds=0.1,
                ),
            )
        )
        assert scraper.finish_reason == "scroll_limit"
    assert len(images) == 4
    snapshots = [data for name, data in progress if name == "crawl_progress"]
    assert snapshots[-1]["scroll_attempts"] == 2
    assert snapshots[-1]["scrolls"] > 2
    assert all(data["scroll_attempts"] == 0 for data in snapshots if data["new_images"] > 0)


def test_scroll_movement_alone_resets_attempts(crawl_site, chromium_available):
    base, _ = crawl_site
    progress = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/tall-feed",
                True,
                "#feed",
                lambda name, data: progress.append(data) if name == "crawl_progress" else None,
                CrawlOptions(
                    page_type="infinite",
                    scroll_selector="#feed",
                    max_scrolls=1,
                    idle_rounds=10,
                    wait_seconds=0.1,
                ),
            )
        )
        assert scraper.finish_reason == "scroll_limit"
    assert len(images) == 1
    assert progress[-1]["scrolls"] > 1
    assert progress[-1]["scroll_attempts"] == 1
    assert any(
        data["scrolls"] > 0 and data["new_images"] == 0 and data["scroll_attempts"] == 0
        for data in progress
    )


def test_new_images_reset_failed_attempts_before_retry_limit(
    crawl_site, chromium_available, monkeypatch
):
    base, _ = crawl_site
    attempts = 0
    original = PageScraper.scroll_state

    def inject_delayed_image(page, selector, scroll=False):
        nonlocal attempts
        if scroll:
            attempts += 1
            if attempts == 2:
                page.evaluate(
                    "document.body.insertAdjacentHTML('beforeend', '<img src=\"/1.png\">')"
                )
        return original(page, selector, scroll)

    monkeypatch.setattr(PageScraper, "scroll_state", staticmethod(inject_delayed_image))
    progress = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/null-scroller",
                True,
                "",
                lambda name, data: progress.append(data) if name == "crawl_progress" else None,
                CrawlOptions(page_type="infinite", max_scrolls=2, idle_rounds=10, wait_seconds=0.1),
            )
        )
        assert scraper.finish_reason == "scroll_limit"
    assert len(images) == 2
    fresh_batch = next(
        index
        for index, data in enumerate(progress)
        if data["scrolls"] > 0 and data["new_images"] > 0
    )
    assert progress[fresh_batch - 1]["scroll_attempts"] == 1
    assert progress[fresh_batch]["scroll_attempts"] == 0
    assert progress[-1]["scroll_attempts"] == 2
    assert attempts == 4


def test_real_browser_pagination_buttons(crawl_site, chromium_available):
    base, _ = crawl_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/buttons",
                True,
                "#listing",
                lambda *_: None,
                CrawlOptions(
                    page_type="pagination", next_selector="#next", idle_rounds=1, wait_seconds=0.1
                ),
            )
        )
        assert scraper.finish_reason == "no_next_page"
    assert [image.url for image in images] == [f"{base}/{index}.png" for index in range(3)]


def test_real_browser_repeating_button_stops(crawl_site, chromium_available):
    base, _ = crawl_site
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/repeating-button",
                True,
                "",
                lambda *_: None,
                CrawlOptions(page_type="pagination", idle_rounds=1, wait_seconds=0.1),
            )
        )
        assert scraper.finish_reason == "repeated_page"
    assert len(images) == 1


@pytest.mark.parametrize("selector", ["", ".missing-container"])
def test_real_browser_null_scroller_and_missing_selector_fall_back(
    crawl_site, chromium_available, selector
):
    base, _ = crawl_site
    events = []
    with PageScraper(threading.Event()) as scraper:
        images = list(
            scraper.iter_sources(
                base + "/null-scroller",
                True,
                "",
                lambda name, data: events.append((name, data)),
                CrawlOptions(
                    page_type="infinite", scroll_selector=selector, idle_rounds=2, wait_seconds=0.1
                ),
            )
        )
        assert scraper.finish_reason == "no_new_images"
    assert len(images) == 1
    warnings = [data for name, data in events if name == "log" and "did not match" in str(data)]
    assert len(warnings) == (1 if selector else 0)
    if selector:
        assert selector in warnings[0]
        assert "whole page" in warnings[0]
