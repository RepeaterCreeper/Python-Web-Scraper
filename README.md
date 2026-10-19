# SScraper

For matching faces in an existing local image folder, use the separate
[Folder Face Finder app](folder_face_finder/README.md):
`.venv/bin/python folder_face_finder/run.py`.

A Python desktop app that downloads images from **non-explicit websites** and
optionally compares detected faces with one or more reference photos using
[DeepFace](https://github.com/serengil/deepface). Enter the page URL directly in the app.

## Run

Python 3.11-3.15 is supported for scraping and CSV viewing. Face matching requires
Python 3.11-3.13; TensorFlow does not yet provide Python 3.14/3.15 builds.
On those newer versions, leave reference photos empty to download images only.

On Windows with Python 3.15:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,browser]"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m sscraper
```

The project environment is already installed in `.venv` in this workspace:

```sh
.venv/bin/python -m sscraper
```

For a fresh installation, use Python 3.11 or 3.12:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m sscraper
```

Optional browser mode renders JavaScript and scrolls to trigger lazy images:

```sh
python -m pip install -e ".[browser]"
python -m playwright install chromium
```

Alternatively, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra browser
uv run --extra browser python -m playwright install chromium
uv run --extra browser sscraper
```

## Use

1. Enter a complete `https://` or `http://` page URL and choose how the site loads
   more images: **Single page**, **Infinite scroll**, or **Pagination / Next page**.
2. **Leave reference photos empty to download images only.** DeepFace is not loaded
   in this mode. Images are saved locally and listed with a `downloaded` status;
   source links and file paths are recorded in the CSV.
   For face matching, add one or more clear photos of **the same person**, each containing exactly one
   face. Use different angles and lighting; crop out other people. Every reference
   is validated before scanning starts.
3. For face matching, keep the default SFace model and OpenCV detector to start.
   Enable **Render JavaScript** if images are inserted by scripts or pagination
   uses a JavaScript button. Infinite scroll enables browser rendering automatically.
4. Choose an output folder and click **Start scan**.
5. Previews are hidden by default. Enable **Show previews** to display live download
   and face-comparison previews and result thumbnails. A box marks the closest
   detected face after comparison. Toggle **Hide URL** to mask the Page URL input
   like a password field. Both choices are remembered and can be changed during a scan.
6. Filter to **Candidates only**, select a result to open its source page, or
   double-click a row to open the downloaded image. Open the scan folder for the report.

The network scanner and recognition worker operate concurrently. A bounded queue
keeps downloading from outrunning inference. **Stop** cancels pending work and
preserves results already written; the current network/model operation must
finish before the worker shuts down. Closing the app during a scan also requests
a graceful stop.

Multiple references are embedded once per run. Each detected face in a downloaded
image is compared against all reference embeddings with DeepFace's cosine metric.
The closest comparison supplies the reported distance and the model's default
threshold. A candidate match means **at least one reference passed the threshold**;
the `Refs` column shows how many references passed for that same face. It is not
a calibrated probability or a confirmed identity. Review candidates yourself.
Small, blurred, obscured, or profile faces can be missed.

## View results as thumbnail cards

Click **View CSV cards…** beside Results to open the current scan's CSV, or choose
a previous `results.csv`. You can also launch the viewer independently:

```sh
.venv/bin/python -m sscraper.csv_viewer
# Or supply a CSV path directly:
.venv/bin/python -m sscraper.csv_viewer /path/to/results.csv
```

Each non-error row becomes a card with a local thumbnail, title, status, optional
match distance, **Open image**, and **Open source** buttons. **Open source** uses
`link_url` first (usually the watch/detail page), then `page_url`. It does not
extract a direct video stream. Rows whose status indicates an error/failure, or
whose `error` column is nonempty, are excluded. Successful no-face and no-match
rows remain visible. Search, status filters and pages of 24 cards support larger
reports. **Reload** refreshes a report while its scan is still running.

Click any thumbnail, or **Single image**, for a large image view. Use **Left** and
**Right** arrow keys (or Previous/Next) to move through the entire current filtered
result set, including across card pages. **Escape** or **Back to cards** returns to
the card page containing the current image. Source/image links remain available
in single-image mode. Navigation stops at the first/last image; missing files show
a placeholder and do not prevent navigating onward. When typing in Search, arrow
keys keep their normal text-editing behavior.

### Bookmarks, labels and tags

Use the **☆ / ★** button on a card to bookmark it, or **Edit…** to set a personal
label and comma-separated tags. Single-image view has the same fields, a Save
button, and the **B** shortcut for toggling bookmarks. Labels/tags autosave when
you press Enter, leave a field, navigate away, or close the viewer. Arrow and B
shortcuts are disabled while typing in a text field. Personal labels become the
display title; the original scraped `label` column is preserved.

On the first edit the viewer creates **`results.annotated.csv`** beside the input
CSV. Edits are saved automatically to that copy using atomic file replacement.
The file being viewed is never modified. If that copy's name already exists, a
new name such as `results.annotated-2.csv` is used; existing files are not overwritten.
The copy preserves all source rows and columns, including hidden error rows and
custom columns, and adds **`bookmarked`**, **`review_label`**, and **`tags`**.
Annotations are attached to individual CSV rows, so repeated filenames remain
independently reviewable. Tags are separated by commas and duplicate tags are
removed without regard to capitalization.

Use **Bookmarked** to filter cards, and Search to find labels or tags. Open an
annotated copy to restore saved review data. Further edits create a new copy of
that file as well. Reload uses the current annotated copy once edits have been
saved. A copy is a snapshot of the loaded report; it does not automatically add
new rows from a scraper that is still running. Keep it beside its images for
portable previews. No CSV copy is created when you only browse images.

The viewer supports both scraper CSVs and Folder Face Finder reports. It reads
local previews and does not automatically fetch remote images. If an image is
missing, a placeholder is shown; **Open image** can still open its HTTP(S) image URL
if one is available. Moving an entire scraper run folder is supported by looking
for the stored filename under the CSV's adjacent `images/` folder. Keep images
with their CSV for portable previews. No face-recognition models are loaded by
the standalone viewer.

## Image discovery

By default, the app discovers image URLs in `img`, `picture`/`source`, `srcset`,
common lazy-loading attributes, video posters, social-image metadata, and inline
CSS backgrounds. All `srcset` variants are included. URLs are deduplicated and
resolved against the page's final URL and any HTML `base` element. Images served
from a CDN are supported. Inline raster data images are supported too.

Browser mode additionally discovers current image sources and computed CSS
backgrounds, including pseudo-elements, and transfers browser cookies to the
image-download session. Images are collected and downloaded throughout discovery,
including before each scroll, rather than only from the final page snapshot.
URLs are deduplicated across the whole run. An optional CSS selector restricts
discovery to matching elements and their descendants.

### Page types and stopping conditions

| Page type | Behavior | Natural stopping condition |
| --- | --- | --- |
| Single page | Read the initial HTML. Browser rendering also attempts up to 12 scrolls for lazy images. | The page's discovered images have been handled. Browser scrolling may finish earlier after repeated idle attempts at the bottom. |
| Infinite scroll | Keep the browser page open and collect images after every scroll. | By default, 3 consecutive attempts at the bottom make no progress. |
| Pagination / Next page | Follow a Next link, or click a Next button with browser rendering enabled. | Next is missing/disabled, a page URL repeats, or a JavaScript button repeats a listing's image URLs. |

Choose the page type explicitly using **How does this site load more images?**
The app remembers your selection. For pagination, leaving **Next link / button
CSS selector** empty detects common `rel="next"`, `.next`, accessible Next labels,
and English Next text or arrow controls. Enter a site-specific selector, such as
`a.next-page` or `button#next`, when detection is ambiguous. Non-English or numbered
pagination generally needs a selector. Requests mode requires a real link; browser
mode also supports JavaScript buttons and listings that change without a URL change.
Next links stay on the starting site's final host; image CDN URLs can still be external.

### Parallel pagination with URL parameters

Choose **Pagination / Next page**, then enter a **Pagination URL parameter**
when the site supports numbered pages or offsets. This generates URLs directly;
no Next link is required. For example:

| Setting | Numbered pages | Offset-based listings |
| --- | --- | --- |
| Page URL | `https://example.com/gallery?category=cats` | `https://example.com/gallery` |
| Pagination URL parameter | `page` | `offset` |
| Starting page / offset | `1` | `0` |
| Page / offset increment | `1` | `20` |
| Extra query parameters | `sort=newest` | `limit=20&sort=newest` |
| Maximum pages | `10` | `10` |
| Parallel page workers | `4` | `4` |

The first example requests `?category=cats&sort=newest&page=1` through `page=10`.
The second requests offsets `0, 20, …, 180`. **Maximum pages** is a request count,
not the final page number. Existing URL parameters are preserved; extra parameters
replace existing values with the same name. The generated page/offset value takes
precedence over both. Enter extras as a URL query string, using percent encoding
for literal `&` or `=` within values.

Up to **Parallel page workers** pages are fetched together (default 4, maximum 16).
Results are consumed in page order and image URLs are deduplicated across pages.
**Wait for new content / between pages** sets the pause between batches; in browser
mode it also controls rendering waits. Browser rendering uses an independent browser
per page, so fewer workers may be appropriate for memory-heavy sites. Image downloads
and face processing retain their existing bounded pipeline.

Discovery stops at the page/image/time limit, an empty image listing, a repeated
listing or final URL, or a page error. Set the image selector to the listing to avoid
unrelated header/footer images affecting end detection. Pages already in the current
batch may have been requested when a stop is detected; no further batches are started.
Stop and early iterator closure cancel pending work and wait for active network/browser
operations to clean up. HTTP errors are reported, and previously saved results remain.

Leave **Pagination URL parameter** blank to use the existing Next-link/button flow.
That flow is sequential because the next target depends on the current page.
All new controls are remembered and saved in each run's `config.json`.

For feeds that scroll inside a panel rather than the window, enter the panel's
selector under **Scroll container CSS selector**. Leave it blank for the whole page.
If the selector does not match an element, the app logs a warning and continues
with whole-page scrolling; a feed inside a panel still needs its correct selector.
Increase **Wait for new content / between pages** and the number of idle attempts
for sites that load slowly. The default wait is 1.5 seconds. Idle counts reset when
new image URLs appear or scrolling moves forward; attempts above the bottom do not count as end-of-feed evidence.

Additional stopping limits are configurable: **100 pages**, **500 consecutive scroll attempts without progress**,
and **30 minutes of discovery** by default. The global image limit counts unique
image URLs attempted across all pages and batches, including failed downloads.
Zero disables the image limit; zero minutes disables the time limit. Time is
checked between discovery operations, so an ongoing request, download, or queued
model operation can finish after that time. Already queued images are processed
after discovery reaches a limit. **Stop** cancels the remaining queue instead.

The scroll-attempt limit resets to zero whenever scrolling moves forward or finds
new image URLs. Successful scrolling can continue beyond 500 total scrolls.
The live status shows pages visited, consecutive attempts without progress, total
scrolls, and the discovery stop reason.
These are also recorded in `summary.json`. A limit ending discovery is distinct
from a scan error. Previously completed downloads and report rows are preserved.

This is a general page-image scraper, not a promise to extract every possible
image from every website: login-only resources, bot challenges, iframes, canvas,
blob URLs, SVG/vector assets, feeds that require a Load More button instead of scrolling,
and unusual lazy-loading formats may need a site-specific adapter. The app does
not automate login or bypass access controls. HTTP redirects are followed normally.

Downloads use timeouts, a 20 MB per-image limit, a 10 MB HTML limit, a 25-megapixel
decoded-image limit, and an adjustable delay. Images are normalized to RGB JPEG
with EXIF orientation applied for consistent preview and inference. Non-raster,
corrupt, oversized, or inaccessible resources are logged and skipped. No-face
results are distinguished from download and recognition errors.

## Output

Each scan creates a separate timestamped folder under the chosen output folder:

```text
runs/20261014-120000-<id>/
  config.json       # URL, model, detector, options and local reference paths
  results.csv       # Per-image status, links, distance, threshold and errors
  summary.json      # Completion/stop outcome and counts
  events.jsonl      # Timestamped events and diagnostic tracebacks
  images/           # Downloaded raster images, named by content hash
```

Results are written incrementally. Identical downloaded files share a local file,
while separate source URLs still receive their own report rows. Stopped runs may
have downloaded images that were not processed; `summary.json` records both the
queued and total unprocessed counts. Reference embeddings stay in memory.
Model weights are cached by DeepFace,
usually under `~/.deepface/weights`, and may download on first use. Comparisons run
locally; this app does not upload reference photos to a recognition API.

## Development checks

```sh
python -m pip install -e ".[dev,browser]"
python -m pytest -q
ruff check sscraper tests
```

Tests cover HTML discovery, URL validation, inline-image decoding, reference
validation, multiple faces/references, threshold overrides, a real local HTTP
scraping pipeline, error reports, and cancellation under queue backpressure.
They also cover pagination cycles, global image/page/time limits, cross-site Next
links, and real Chromium tests for incremental infinite scrolling, JavaScript
Next buttons, repeated listings, and scroll limits. Browser tests skip if
Playwright or Chromium is not installed. Face inference is stubbed in automated
tests so they run without recognition-model downloads.
