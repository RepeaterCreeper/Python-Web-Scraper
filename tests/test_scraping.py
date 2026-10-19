import base64
import io
import threading

import pytest
from PIL import Image

from sscraper.scraping import Cancelled, ImageSource, PageScraper, extract_sources, validate_url


def test_extracts_lazy_picture_poster_background_and_metadata():
    html = """<base href="/assets/">
    <a href="/people/alex"><img alt="Alex" src="small.jpg" data-src="full.jpg"
       srcset="small.jpg 100w, large.jpg 500w"></a>
    <picture><source srcset="wide.webp 2x"><img src="full.jpg"></picture>
    <video poster="poster.jpg"><source src="movie.mp4"></video>
    <div style="background-image:url('background.jpg')"></div>
    <meta property="og:image" content="social.jpg">
    <img src="javascript:alert(1)">
    """
    sources = extract_sources(html, "https://example.com/gallery")
    assert {s.url for s in sources} == {
        f"https://example.com/assets/{name}"
        for name in (
            "small.jpg",
            "full.jpg",
            "large.jpg",
            "wide.webp",
            "poster.jpg",
            "background.jpg",
            "social.jpg",
        )
    }
    assert sources[0].link_url == "https://example.com/people/alex"
    assert sources[0].label == "Alex"
    assert sources[0].page_url == "https://example.com/gallery"


def test_selector_limits_to_selected_element_and_descendants():
    sources = extract_sources(
        '<img src="outside.jpg"><div class="gallery"><img src="inside.jpg"></div>',
        "https://example.com",
        ".gallery",
    )
    assert [s.url for s in sources] == ["https://example.com/inside.jpg"]
    assert (
        len(
            extract_sources('<img class="target" src="a.jpg">', "https://example.com", "img.target")
        )
        == 1
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/a",
        "example.com",
        "ftp://example.com",
        "https://a:b@example.com",
        "http://example.com:invalid",
    ],
)
def test_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_normalizes_url_fragment():
    assert (
        validate_url(" https://example.com/gallery?q=1#images ")
        == "https://example.com/gallery?q=1"
    )


def test_decodes_inline_images(tmp_path):
    image = Image.new("RGB", (30, 20), "red")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    sources = extract_sources(f'<img src="{uri}">', "https://example.com")
    with PageScraper(threading.Event()) as scraper:
        downloaded = scraper.download(sources[0], tmp_path)
    assert downloaded.path.is_file()
    assert (downloaded.width, downloaded.height) == (30, 20)


def test_stop_prevents_network_request(tmp_path):
    stop = threading.Event()
    stop.set()
    with PageScraper(stop) as scraper, pytest.raises(Cancelled):
        scraper.download(ImageSource("https://example.com/a.jpg", "", "", ""), tmp_path)
