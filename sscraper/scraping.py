from __future__ import annotations

import base64
import hashlib
import io
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    unquote_to_bytes,
    urldefrag,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

USER_AGENT = "SScraper/0.1 (image scanner)"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PAGE_BYTES = 10 * 1024 * 1024
# Prevent oversized decoded images from consuming the recognition worker's memory.
Image.MAX_IMAGE_PIXELS = 25_000_000


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class ImageSource:
    url: str
    page_url: str
    link_url: str
    label: str


@dataclass(frozen=True)
class DownloadedImage:
    source: ImageSource
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CrawlOptions:
    page_type: str = "single"
    next_selector: str = ""
    max_pages: int = 100
    max_scrolls: int = 500  # Consecutive attempts without forward movement or new images.
    idle_rounds: int = 3
    wait_seconds: float = 1.5
    max_seconds: float = 1800
    scroll_selector: str = ""
    page_param: str = ""
    page_start: int = 1
    page_step: int = 1
    page_workers: int = 4
    extra_params: str = ""


def parameter_page_url(url: str, options: CrawlOptions, index: int) -> str:
    """Merge query values, retaining repeated and blank unrelated parameters."""
    parsed = urlsplit(url)
    extras = parse_qsl(options.extra_params.lstrip("?"), keep_blank_values=True)
    overridden = {key for key, _ in extras} | {options.page_param}
    pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in overridden
    ]
    pairs.extend((key, value) for key, value in extras if key != options.page_param)
    pairs.append((options.page_param, str(options.page_start + index * options.page_step)))
    return urlunsplit(parsed._replace(query=urlencode(pairs), fragment=""))


def next_page_url(html: str, page_url: str, selector: str = "") -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    nodes = (
        soup.select(selector)
        if selector
        else soup.select(
            'a[rel~="next"], link[rel~="next"], a.next, .pagination a.next, a[aria-label]'
        )
    )
    if not selector:
        nodes = [
            node
            for node in nodes
            if node.name == "link"
            or "next" in node.get("rel", [])
            or "next" in node.get("class", [])
            or re.search(r"\bnext\b", node.get("aria-label", ""), re.I)
        ]
        nodes += [
            node
            for node in soup.find_all("a", href=True)
            if node.get_text(" ", strip=True).lower()
            in {"next", "next page", "next ›", "next »", "›", "»", "→"}
        ]
    for node in nodes:
        if (
            node.get("aria-disabled") == "true"
            or node.has_attr("disabled")
            or "disabled" in node.get("class", [])
        ):
            continue
        if node.get("href"):
            base = soup.find("base", href=True)
            base_url = urljoin(page_url, base["href"]) if base else page_url
            raw = node["href"].strip()
            if raw.startswith("#") or not raw:
                continue
            url = urldefrag(urljoin(base_url, raw))[0]
            if urlsplit(url).scheme in {"http", "https"}:
                return url
        elif selector:
            raise ValueError(
                "The Next selector does not point to a link. Enable browser rendering for buttons."
            )
    return None


def validate_url(value: str) -> str:
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete http:// or https:// page URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing login credentials are not supported.")
    # Reading port also validates malformed port numbers.
    _ = parsed.port
    return urldefrag(value)[0]


def extract_sources(html: str, page_url: str, selector: str = "") -> list[ImageSource]:
    """Find embedded raster image URLs, including lazy images, posters and CSS backgrounds.

    All srcset URLs are included, rather than assuming the browser's chosen size is sufficient.
    A selector restricts discovery to the selected elements and their descendants.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_node = soup.find("base", href=True)
    base_url = urljoin(page_url, base_node["href"]) if base_node else page_url
    nodes = soup.select(selector) if selector.strip() else [soup]
    found: dict[str, ImageSource] = {}

    def add(raw: str, node) -> None:
        raw = str(raw).strip()
        if not raw:
            return
        url = raw if raw.startswith("data:image/") else urldefrag(urljoin(base_url, raw))[0]
        if urlsplit(url).scheme not in {"http", "https", "data"}:
            return
        anchor = node if node.name == "a" else node.find_parent("a", href=True)
        link = urljoin(base_url, anchor.get("href", "")) if anchor else page_url
        label = str(node.get("alt") or node.get("title") or node.get("aria-label") or "")
        found.setdefault(url, ImageSource(url, page_url, link, label))

    def srcset(raw: str, node) -> None:
        # A data URI contains commas and represents one image; don't split its payload.
        if raw.strip().startswith("data:"):
            add(raw.strip().split()[0], node)
            return
        for entry in raw.split(","):
            parts = entry.strip().split()
            if parts:
                add(parts[0], node)

    for root in nodes:
        for node in [root, *root.find_all(True)]:
            if node.name in {"img", "source", "input"}:
                if node.name == "input" and node.get("type") != "image":
                    continue
                # <source> inside audio/video describes media files, not images.
                if node.name == "source" and not node.find_parent("picture"):
                    continue
                for attr in ("src", "data-src", "data-original", "data-lazy-src", "data-url"):
                    if node.get(attr):
                        add(node[attr], node)
                for attr in ("srcset", "data-srcset", "data-lazy-srcset"):
                    if node.get(attr):
                        srcset(node[attr], node)
            if node.name == "video" and node.get("poster"):
                add(node["poster"], node)
            if node.name == "meta" and node.get("property", node.get("name")) in {
                "og:image",
                "og:image:url",
                "twitter:image",
                "twitter:image:src",
            }:
                add(node.get("content", ""), node)
            for match in re.finditer(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", node.get("style", "")):
                add(match.group(1), node)
    return list(found.values())


def read_response(response, limit: int, stop: threading.Event) -> bytes:
    response.raise_for_status()
    if int(response.headers.get("Content-Length", "0") or "0") > limit:
        raise ValueError("Resource exceeds the download size limit.")
    chunks, size = [], 0
    deadline = time.monotonic() + 60
    for chunk in response.iter_content(64 * 1024):
        if stop.is_set():
            raise Cancelled()
        if time.monotonic() > deadline:
            raise TimeoutError("Download exceeded 60 seconds.")
        size += len(chunk)
        if size > limit:
            raise ValueError("Resource exceeds the download size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


class PageScraper:
    """Own one network session in the scanning thread."""

    def __init__(self, stop: threading.Event):
        self.stop = stop
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.finish_reason = ""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.session.close()

    def check_stop(self, started: float, options: CrawlOptions) -> bool:
        if self.stop.is_set():
            raise Cancelled()
        if options.max_seconds and time.monotonic() - started >= options.max_seconds:
            self.finish_reason = "time_limit"
            return True
        return False

    def fetch_html(self, url: str) -> tuple[str, str]:
        with self.session.get(url, timeout=(10, 20), stream=True) as response:
            raw = read_response(response, MAX_PAGE_BYTES, self.stop)
            return raw.decode(response.encoding or "utf-8", errors="replace"), response.url

    def page(self, url: str, browser: bool, selector: str, emit) -> list[ImageSource]:
        """Compatibility helper for a single page; scanning uses the streaming iterator."""
        return list(self.iter_sources(url, browser, selector, emit, CrawlOptions()))

    def iter_sources(self, url: str, browser: bool, selector: str, emit, options: CrawlOptions):
        if options.page_type not in {"single", "infinite", "pagination"}:
            raise ValueError("Choose Single page, Infinite scroll, or Pagination.")
        if (
            min(options.max_pages, options.max_scrolls, options.idle_rounds) < 1
            or options.wait_seconds <= 0
            or options.max_seconds < 0
        ):
            raise ValueError(
                "Page, scroll and idle limits must be positive, with a positive wait and nonnegative time limit."
            )
        started = time.monotonic()
        seen: set[str] = set()
        self.finish_reason = ""
        if options.page_type == "pagination" and options.page_param:
            if (
                options.page_start < 0
                or options.page_step < 1
                or not 1 <= options.page_workers <= 16
            ):
                raise ValueError("Page start must be nonnegative, step positive, and workers 1–16.")
            yield from self.iter_parameter_pages(url, browser, selector, emit, options, started)
            return
        if not browser and options.page_type != "infinite":
            visited = set()
            origin = None
            current = url
            for index in range(options.max_pages):
                if self.check_stop(started, options):
                    return
                if current in visited:
                    self.finish_reason = "repeated_page"
                    return
                emit("scan_status", f"Loading page {index + 1}: {current}")
                html, final_url = self.fetch_html(current)
                if origin is None:
                    origin = urlsplit(final_url).netloc.lower()
                elif urlsplit(final_url).netloc.lower() != origin:
                    self.finish_reason = "next_page_outside_site"
                    return
                if final_url in visited:
                    self.finish_reason = "repeated_page"
                    return
                visited.update((current, final_url))
                if self.check_stop(started, options):
                    return
                sources = extract_sources(html, final_url, selector)
                fresh = [source for source in sources if source.url not in seen]
                emit(
                    "crawl_progress",
                    {"pages": index + 1, "scrolls": 0, "new_images": len(fresh), "url": final_url},
                )
                for source in fresh:
                    if self.check_stop(started, options):
                        return
                    seen.add(source.url)
                    yield source
                if options.page_type == "single":
                    self.finish_reason = "single_page_complete"
                    return
                current = next_page_url(html, final_url, options.next_selector)
                if not current:
                    self.finish_reason = "no_next_page"
                    return
                if urlsplit(current).netloc.lower() != origin:
                    self.finish_reason = "next_page_outside_site"
                    return
                if self.stop.wait(options.wait_seconds):
                    raise Cancelled()
            self.finish_reason = "page_limit"
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                'Browser mode needs: pip install -e ".[browser]" '
                "and python -m playwright install chromium"
            ) from exc
        with sync_playwright() as playwright:
            browser_instance = playwright.chromium.launch(headless=True)
            try:
                context = browser_instance.new_context()
                page = context.new_page()
                self.session.headers["User-Agent"] = page.evaluate("navigator.userAgent")
                current = url
                origin = None
                visited: set[str] = set()
                signatures: set[tuple[str, ...]] = set()
                click_next = False
                total_scrolls = 0
                scroll_warnings: set[str] = set()
                for page_index in range(options.max_pages):
                    if self.check_stop(started, options):
                        return
                    if not click_next:
                        if current in visited:
                            self.finish_reason = "repeated_page"
                            return
                        emit("scan_status", f"Loading page {page_index + 1}: {current}")
                        response = page.goto(current, wait_until="domcontentloaded", timeout=30_000)
                        if response and response.status >= 400:
                            raise RuntimeError(f"Page returned HTTP {response.status}.")
                    self.browser_wait(page, options.wait_seconds, started, options)
                    if self.check_stop(started, options):
                        return
                    final_url = urldefrag(page.url)[0]
                    if origin is None:
                        origin = urlsplit(final_url).netloc.lower()
                    elif urlsplit(final_url).netloc.lower() != origin:
                        self.finish_reason = "next_page_outside_site"
                        return
                    if not click_next and final_url in visited:
                        self.finish_reason = "repeated_page"
                        return
                    visited.update((current, final_url))
                    sources = self.browser_sources(page, context, selector)
                    signature = tuple(sorted(source.url for source in sources))
                    # JS Next buttons may replace the listing without changing the URL.
                    if click_next and signature in signatures:
                        self.finish_reason = "repeated_page"
                        return
                    signatures.add(signature)
                    idle = 0
                    scroll_count = 0
                    scroll_attempts = 0
                    while True:
                        if self.check_stop(started, options):
                            return
                        fresh = [source for source in sources if source.url not in seen]
                        emit(
                            "crawl_progress",
                            {
                                "pages": page_index + 1,
                                "scrolls": total_scrolls,
                                "scroll_attempts": scroll_attempts,
                                "new_images": len(fresh),
                                "url": page.url,
                            },
                        )
                        for source in fresh:
                            if self.check_stop(started, options):
                                return
                            seen.add(source.url)
                            yield source
                        if options.page_type != "infinite":
                            # Also collect while moving through ordinary long pages with lazy images.
                            if scroll_count >= 12:
                                break
                        emit(
                            "scan_status",
                            f"Scrolling page {page_index + 1} · attempts without progress {scroll_attempts}",
                        )
                        state = self.scroll_state(page, options.scroll_selector, scroll=True)
                        if (
                            state["missing_selector"]
                            and options.scroll_selector not in scroll_warnings
                        ):
                            scroll_warnings.add(options.scroll_selector)
                            emit(
                                "log",
                                f"Scroll container {options.scroll_selector!r} did not match an element. "
                                "Continuing with the whole page. Clear the Scroll container field, "
                                "or enter the correct selector if the feed scrolls inside a panel.",
                            )
                        self.browser_wait(page, options.wait_seconds, started, options)
                        if self.check_stop(started, options):
                            return
                        scroll_count += 1
                        total_scrolls += 1
                        sources = self.browser_sources(page, context, selector)
                        new_count = sum(source.url not in seen for source in sources)
                        after_scroll = self.scroll_state(page, options.scroll_selector)
                        progressed = (
                            new_count > 0
                            or state["moved"]
                            or after_scroll["position"] > state["position"] + 0.5
                        )
                        scroll_attempts = 0 if progressed else scroll_attempts + 1
                        at_bottom = after_scroll["at_bottom"]
                        idle = idle + 1 if not progressed and at_bottom else 0
                        if new_count == 0:
                            self_event = {
                                "pages": page_index + 1,
                                "scrolls": total_scrolls,
                                "scroll_attempts": scroll_attempts,
                                "new_images": 0,
                                "url": page.url,
                            }
                            emit("crawl_progress", self_event)
                        if (
                            options.page_type == "infinite"
                            and scroll_attempts >= options.max_scrolls
                        ):
                            self.finish_reason = "scroll_limit"
                            emit(
                                "log",
                                f"Stopped after {scroll_attempts} consecutive scroll attempts without progress.",
                            )
                            return
                        if options.page_type == "infinite" and idle >= options.idle_rounds:
                            self.finish_reason = "no_new_images"
                            emit(
                                "log",
                                f"Stopped after {idle} attempts at the bottom found no new images.",
                            )
                            return
                        if options.page_type != "infinite" and idle >= options.idle_rounds:
                            break
                    if options.page_type == "single":
                        self.finish_reason = "single_page_complete"
                        return
                    control = self.next_control(page, options.next_selector)
                    if control is None:
                        self.finish_reason = "no_next_page"
                        return
                    href = control.get_attribute("href")
                    next_url = (
                        urldefrag(urljoin(page.url, href))[0]
                        if href and not href.startswith("#")
                        else None
                    )
                    if next_url and urlsplit(next_url).scheme in {"http", "https"}:
                        if urlsplit(next_url).netloc.lower() != origin:
                            self.finish_reason = "next_page_outside_site"
                            return
                        current, click_next = next_url, False
                    else:
                        emit("scan_status", "Clicking Next…")
                        control.click(timeout=10_000)
                        page.wait_for_load_state("domcontentloaded", timeout=30_000)
                        click_next = True
                        current = urldefrag(page.url)[0]
                self.finish_reason = "page_limit"
            finally:
                browser_instance.close()

    def iter_parameter_pages(self, url, browser, selector, emit, options, started):
        """Fetch a bounded batch concurrently and consume it in page order.

        Each task owns its session/browser. Only the coordinator emits progress,
        merges cookies, and deduplicates results. Closing the iterator cancels work.
        """
        worker_stop = threading.Event()
        seen = set()
        visited = set()
        signatures = set()
        origin = None
        executor = ThreadPoolExecutor(max_workers=options.page_workers, thread_name_prefix="pages")
        futures = []

        def fetch(target, headers, cookies):
            with PageScraper(worker_stop) as worker:
                worker.session.headers.update(headers)
                worker.session.cookies.update(cookies)
                if worker_stop.is_set():
                    raise Cancelled()
                final_url = target
                if browser:

                    def capture(name, data):
                        nonlocal final_url
                        if name == "crawl_progress":
                            final_url = data["url"]

                    remaining = (
                        max(1e-9, options.max_seconds - (time.monotonic() - started))
                        if options.max_seconds
                        else 0
                    )
                    sources = list(
                        worker.iter_sources(
                            target,
                            True,
                            selector,
                            capture,
                            replace(options, page_type="single", max_seconds=remaining),
                        )
                    )
                else:
                    html, final_url = worker.fetch_html(target)
                    sources = extract_sources(html, final_url, selector)
                return (
                    sources,
                    final_url,
                    worker.session.cookies.copy(),
                    dict(worker.session.headers),
                )

        try:
            for batch_start in range(0, options.max_pages, options.page_workers):
                if self.check_stop(started, options):
                    return
                futures = []
                for index in range(
                    batch_start, min(batch_start + options.page_workers, options.max_pages)
                ):
                    target = parameter_page_url(url, options, index)
                    emit("scan_status", f"Loading page {index + 1}: {target}")
                    futures.append(
                        executor.submit(
                            fetch, target, dict(self.session.headers), self.session.cookies.copy()
                        )
                    )
                for offset, future in enumerate(futures):
                    while True:
                        if self.check_stop(started, options):
                            return
                        try:
                            sources, final_url, cookies, headers = future.result(timeout=0.1)
                            break
                        except TimeoutError:
                            if future.done():
                                raise
                    if self.check_stop(started, options):
                        return
                    host = urlsplit(final_url).netloc.lower()
                    if origin is None:
                        origin = host
                    elif host != origin:
                        self.finish_reason = "next_page_outside_site"
                        return
                    signature = tuple(sorted(source.url for source in sources))
                    if final_url in visited or (signature and signature in signatures):
                        self.finish_reason = "repeated_page"
                        return
                    visited.add(final_url)
                    signatures.add(signature)
                    self.session.cookies.update(cookies)
                    self.session.headers.update(headers)
                    fresh = [source for source in sources if source.url not in seen]
                    emit(
                        "crawl_progress",
                        {
                            "pages": batch_start + offset + 1,
                            "scrolls": 0,
                            "new_images": len(fresh),
                            "url": final_url,
                        },
                    )
                    if not sources:
                        self.finish_reason = "no_new_images"
                        return
                    for source in fresh:
                        if self.check_stop(started, options):
                            return
                        seen.add(source.url)
                        yield source
                if batch_start + options.page_workers < options.max_pages:
                    if self.stop.wait(options.wait_seconds):
                        raise Cancelled()
            self.finish_reason = "page_limit"
        finally:
            worker_stop.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def scroll_state(page, selector: str, scroll: bool = False) -> dict:
        """Use window scrolling when the custom element is absent.

        document.scrollingElement can be null in quirks-mode documents. Window
        scrolling avoids relying on whether body or html owns scrollTop.
        Re-resolve containers every time, since feeds may replace their DOM.
        """
        return page.evaluate(
            """({selector, scroll}) => {
            const selected = selector ? document.querySelector(selector) : null;
            const missing = Boolean(selector && !selected);
            const useWindow = !selected || selected === document.body || selected === document.documentElement;
            if (useWindow) {
                const before = window.scrollY;
                if (scroll) window.scrollBy(0, Math.max(window.innerHeight, 600));
                const height = Math.max(document.documentElement?.scrollHeight || 0,
                                        document.body?.scrollHeight || 0, window.innerHeight);
                return {missing_selector: missing, position: window.scrollY, moved: window.scrollY > before + 0.5,
                        at_bottom: window.scrollY + window.innerHeight >= height - 5};
            }
            const before = selected.scrollTop;
            if (scroll) selected.scrollBy(0, Math.max(selected.clientHeight, 600));
            return {missing_selector: false, position: selected.scrollTop, moved: selected.scrollTop > before + 0.5,
                    at_bottom: selected.scrollTop + selected.clientHeight >= selected.scrollHeight - 5};
        }""",
            {"selector": selector, "scroll": scroll},
        )

    def browser_wait(self, page, seconds: float, started: float, options: CrawlOptions):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if self.check_stop(started, options):
                return
            page.wait_for_timeout(min(100, max(1, (until - time.monotonic()) * 1000)))

    @staticmethod
    def next_control(page, selector: str):
        if selector:
            candidates = page.locator(selector)
        else:
            candidates = page.locator(
                'a[rel~="next"], a.next, button.next, [aria-label*="Next"], [aria-label*="next"]'
            )
        for candidate in candidates.all():
            if (
                candidate.is_visible()
                and candidate.is_enabled()
                and candidate.get_attribute("aria-disabled") != "true"
                and "disabled" not in (candidate.get_attribute("class") or "").split()
            ):
                return candidate
        if not selector:
            for role in ("link", "button"):
                for candidate in page.get_by_role(
                    role, name=re.compile(r"^(next(?: page)?\s*[›»→]?|[›»→])$", re.I)
                ).all():
                    if (
                        candidate.is_visible()
                        and candidate.is_enabled()
                        and candidate.get_attribute("aria-disabled") != "true"
                    ):
                        return candidate
        return None

    def browser_sources(self, page, context, selector: str) -> list[ImageSource]:
        html, final_url = page.content(), page.url
        sources = extract_sources(html, final_url, selector)
        # Computed styles include images from external CSS and browser-selected picture sources.
        extra = page.evaluate(
            """(selector) => {
                    const roots = selector ? [...document.querySelectorAll(selector)] : [document];
                    const nodes = new Set(roots.flatMap(r => [r, ...r.querySelectorAll('*')]));
                    return [...nodes].filter(n => n instanceof Element).flatMap(n => {
                        const urls = [];
                        if (n instanceof HTMLImageElement && n.currentSrc) urls.push(n.currentSrc);
                        for (const pseudo of [null, '::before', '::after']) {
                            const bg = getComputedStyle(n, pseudo).backgroundImage;
                            for (const m of bg.matchAll(/url\\(["']?(.*?)["']?\\)/g)) urls.push(m[1]);
                        }
                        return urls.map(url => ({url, link: n.closest('a')?.href || location.href,
                                                 label: n.getAttribute('alt') || ''}));
                    });
                }""",
            selector,
        )
        seen = {item.url for item in sources}
        for item in extra:
            if item["url"] not in seen and urlsplit(item["url"]).scheme in {
                "http",
                "https",
                "data",
            }:
                sources.append(ImageSource(item["url"], final_url, item["link"], item["label"]))
                seen.add(item["url"])
        for cookie in context.cookies():
            self.session.cookies.set(
                cookie["name"], cookie["value"], domain=cookie["domain"], path=cookie["path"]
            )
        return sources

    def download(self, source: ImageSource, directory: Path) -> DownloadedImage:
        if self.stop.is_set():
            raise Cancelled()
        if source.url.startswith("data:"):
            header, payload = source.url.split(",", 1)
            if len(payload) > MAX_IMAGE_BYTES * 2:
                raise ValueError("Inline image exceeds the size limit.")
            data = base64.b64decode(payload) if ";base64" in header else unquote_to_bytes(payload)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Inline image exceeds the size limit.")
        else:
            with self.session.get(
                source.url, headers={"Referer": source.page_url}, timeout=(10, 20), stream=True
            ) as response:
                data = read_response(response, MAX_IMAGE_BYTES, self.stop)
        # Normalize EXIF orientation and formats so both Qt and DeepFace can use the same local image.
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 25-megapixel limit.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256(data).hexdigest()
            path = directory / f"{digest}.jpg"
            if not path.exists():
                image.save(path, "JPEG", quality=95)
            return DownloadedImage(source, path, image.width, image.height)
