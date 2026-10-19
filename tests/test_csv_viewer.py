import csv

import pytest

from sscraper.csv_viewer import Record, is_web_url, local_image, parse_results


def write_csv(path, rows, fields=None):
    fields = fields or [
        "status",
        "label",
        "local_path",
        "image_url",
        "page_url",
        "link_url",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_parser_skips_errors_and_preserves_successful_rows(tmp_path):
    path = write_csv(
        tmp_path / "results.csv",
        [
            {
                "status": status,
                "local_path": f"{index}.jpg",
                "label": 'A comma, and "quotes"\nUnicode: café',
            }
            for index, status in enumerate(
                [
                    "downloaded",
                    "candidate_match",
                    "no_match",
                    "no_face",
                    "faces_detected",
                    "download_error",
                    "processing_error",
                    "page_error",
                    "failed",
                ]
            )
        ]
        + [{"status": "candidate_match", "local_path": "copy.jpg", "error": "Copy failed"}, {}],
    )
    parsed = parse_results(path)
    assert [record.status for record in parsed.records] == [
        "downloaded",
        "candidate_match",
        "no_match",
        "no_face",
        "faces_detected",
    ]
    assert parsed.records[0].title == 'A comma, and "quotes"\nUnicode: café'
    assert parsed.skipped_errors == 5
    assert parsed.skipped_empty == 1


@pytest.mark.parametrize(
    "contents,message",
    [
        ("", "empty"),
        ("status,label\ndownloaded,photo\n", "containing"),
        ("local_path,local_path\na.jpg,a.jpg\n", "duplicate"),
        ("local_path,status\na.jpg,downloaded,extra\n", "more values"),
    ],
)
def test_invalid_csv_has_a_useful_error(tmp_path, contents, message):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        parse_results(path)


def test_folder_app_report_is_supported(tmp_path):
    path = write_csv(
        tmp_path / "results.csv",
        [
            {
                "status": "faces_detected",
                "source_path": "/photos/alex.jpg",
                "relative_path": "people/alex.jpg",
            }
        ],
        ["status", "source_path", "relative_path"],
    )
    record = parse_results(path).records[0]
    assert record.title == "people/alex.jpg"
    assert record.filename == "alex.jpg"


def test_original_and_relative_local_paths(tmp_path):
    image = tmp_path / "local.jpg"
    image.write_bytes(b"test")
    csv_path = tmp_path / "results.csv"
    assert local_image(Record(2, {"local_path": str(image)}), csv_path) == image
    assert local_image(Record(2, {"local_path": "local.jpg"}), csv_path) == image


@pytest.mark.parametrize("old_path", ["/old/run/images/hash.jpg", r"C:\old\run\images\hash.jpg"])
def test_moved_scraper_folder_recovers_images(tmp_path, old_path):
    (tmp_path / "images").mkdir()
    image = tmp_path / "images" / "hash.jpg"
    image.write_bytes(b"test")
    assert local_image(Record(2, {"local_path": old_path}), tmp_path / "results.csv") == image


def test_folder_app_match_copy_can_supply_preview(tmp_path):
    (tmp_path / "matches" / "nested").mkdir(parents=True)
    image = tmp_path / "matches" / "nested" / "image.jpg"
    image.write_bytes(b"test")
    record = Record(2, {"source_path": "/missing/image.jpg", "relative_path": "nested/image.jpg"})
    assert local_image(record, tmp_path / "results.csv") == image
    assert local_image(Record(2, {"local_path": "/missing.jpg"}), tmp_path / "results.csv") is None


def test_source_link_prefers_watch_page_and_rejects_non_web_urls():
    assert (
        Record(
            2,
            {
                "link_url": "https://example.com/watch/123",
                "page_url": "https://example.com/gallery",
            },
        ).source_url
        == "https://example.com/watch/123"
    )
    assert (
        Record(
            2, {"link_url": "javascript:alert(1)", "page_url": "https://example.com/gallery"}
        ).source_url
        == "https://example.com/gallery"
    )
    assert not is_web_url("file:///tmp/a")
    assert not is_web_url("https:")


def test_interrupted_csv_load_returns_no_partial_cards(tmp_path):
    path = write_csv(tmp_path / "results.csv", [{"local_path": "photo.jpg"}])
    assert parse_results(path, interrupted=lambda: True).records == []


def test_card_grid_scrolls_without_overlap_and_opens_saved_source(tmp_path, monkeypatch):
    from PIL import Image
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication, QPushButton

    from sscraper.csv_viewer import CSVViewer

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "image.png"
    Image.new("RGB", (200, 100), "blue").save(image)
    path = write_csv(
        tmp_path / "results.csv",
        [
            {
                "status": "downloaded",
                "label": f"Photo {index:02d}",
                "local_path": str(image),
                "link_url": f"https://example.com/watch/{index}",
            }
            for index in range(1, 26)
        ]
        + [{"status": "download_error", "error": "404"}],
    )
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True
    )
    viewer = CSVViewer()
    try:
        viewer.show()
        viewer.accept_csv(path, parse_results(path))
        for _ in range(3):
            app.processEvents()
        assert viewer.grid.count() == 24
        assert viewer.parsed.skipped_errors == 1
        first = viewer.grid.itemAtPosition(0, 0).widget()
        below = viewer.grid.itemAtPosition(1, 0).widget()
        assert first.geometry().bottom() < below.geometry().top()
        assert viewer.scroll.verticalScrollBar().maximum() > 0
        button = next(
            button for button in first.findChildren(QPushButton) if button.text() == "Open source"
        )
        button.click()
        assert opened == ["https://example.com/watch/1"]
        viewer.change_page(1)
        assert viewer.grid.count() == 1
        viewer.search.setText("Photo 01")
        assert len(viewer.filtered) == viewer.grid.count() == 1
        assert viewer.page == 0
    finally:
        viewer.close()


def test_single_image_keyboard_navigation_crosses_pages_and_respects_search(tmp_path, monkeypatch):
    from PIL import Image
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from sscraper.csv_viewer import CSVViewer, Thumbnail

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "image.png"
    Image.new("RGB", (200, 100), "blue").save(image)
    path = write_csv(
        tmp_path / "results.csv",
        [
            {
                "status": "downloaded",
                "label": f"Photo {index:02d}",
                "local_path": str(image) if index != 1 else str(tmp_path / "missing.jpg"),
                "link_url": f"https://example.com/watch/{index}",
            }
            for index in range(1, 28)
        ],
    )
    viewer = CSVViewer()
    try:
        viewer.show()
        viewer.activateWindow()
        viewer.accept_csv(path, parse_results(path))
        app.processEvents()
        # Clicking a card enters single-image mode, even when the local file is missing.
        card = viewer.grid.itemAtPosition(0, 0).widget()
        thumbnail = card.findChild(Thumbnail)
        QTest.mouseClick(thumbnail, Qt.MouseButton.LeftButton)
        assert viewer.single_mode
        assert viewer.single_image.text() == "Local image not found"
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Right)
        assert viewer.image_index == 1
        assert not viewer.single_image.original.isNull()
        viewer.show_single(viewer.filtered[23])
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Right)
        assert viewer.image_index == 24
        assert viewer.page_label.text() == "Image 25 / 27"
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Left)
        assert viewer.image_index == 23
        viewer.search.setText("Photo 25")
        assert len(viewer.filtered) == 1
        assert viewer.single_title.text() == "Photo 25"
        viewer.search.setFocus()
        viewer.search.setCursorPosition(len(viewer.search.text()))
        QTest.keyClick(viewer.search, Qt.Key.Key_Left)
        assert viewer.search.cursorPosition() == len(viewer.search.text()) - 1
        assert viewer.image_index == 0
        viewer.search.clear()
        viewer.single_view.setFocus()
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Escape)
        assert not viewer.single_mode
        assert viewer.page == 1
        assert viewer.grid.count() == 3
        viewer.show_single(viewer.filtered[-1])
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Right)
        assert viewer.image_index == 26
        viewer.show_single(viewer.filtered[0])
        QTest.keyClick(viewer.single_view, Qt.Key.Key_Left)
        assert viewer.image_index == 0
    finally:
        viewer.close()


def test_annotations_copy_preserves_original_rows_columns_and_titles(tmp_path):
    from sscraper.csv_viewer import AnnotationStore

    source = write_csv(
        tmp_path / "results.csv",
        [
            {
                "Status": "downloaded",
                "Label": " Original title ",
                "local_path": "same.jpg",
                "custom": "  keep whitespace  ",
            },
            {
                "Status": "no_face",
                "Label": "Second",
                "local_path": "same.jpg",
                "custom": 'Quotes " and comma,',
            },
            {"Status": "download_error", "error": "404", "custom": "retain hidden row"},
            {},
        ],
        ["Status", "Label", "local_path", "custom", "error"],
    )
    original = source.read_bytes()
    parsed = parse_results(source)
    store = AnnotationStore(source, parsed)
    store.save(parsed.records[0], True, "Café, favorite", "Travel, outdoors, travel, , Outdoors")
    copy_path = store.output_path
    assert copy_path.name == "results.annotated.csv"
    assert source.read_bytes() == original
    rows = list(csv.DictReader(copy_path.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 4
    assert list(rows[0]) == [
        "Status",
        "Label",
        "local_path",
        "custom",
        "error",
        "bookmarked",
        "review_label",
        "tags",
    ]
    assert rows[0]["Label"] == " Original title "
    assert rows[0]["custom"] == "  keep whitespace  "
    assert rows[2]["custom"] == "retain hidden row"
    assert rows[0]["tags"] == "Travel, outdoors"
    assert rows[0]["bookmarked"] == "true"
    assert rows[1]["bookmarked"] == ""
    store.save(parsed.records[1], False, "Different row", "second")
    assert store.output_path == copy_path
    restored = parse_results(copy_path)
    assert restored.records[0].bookmarked
    assert restored.records[0].title == "Café, favorite"
    assert restored.records[1].title == "Different row"
    assert source.read_bytes() == original


def test_existing_annotation_copy_is_not_overwritten(tmp_path):
    from sscraper.csv_viewer import AnnotationStore

    source = write_csv(tmp_path / "results.csv", [{"local_path": "image.jpg"}])
    existing = tmp_path / "results.annotated.csv"
    existing.write_text("existing review data")
    parsed = parse_results(source)
    store = AnnotationStore(source, parsed)
    store.save(parsed.records[0], True, "Label", "tag")
    assert store.output_path.name == "results.annotated-2.csv"
    assert existing.read_text() == "existing review data"


def test_opening_annotated_csv_preserves_it_and_reuses_existing_column_names(tmp_path):
    from sscraper.csv_viewer import AnnotationStore

    source = write_csv(
        tmp_path / "review.csv",
        [
            {
                "local_path": "image.jpg",
                "BOOKMARKED": "yes",
                "Review_Label": "Old label",
                "Tags": "old",
            }
        ],
        ["local_path", "BOOKMARKED", "Review_Label", "Tags"],
    )
    original = source.read_bytes()
    parsed = parse_results(source)
    assert parsed.records[0].bookmarked
    store = AnnotationStore(source, parsed)
    store.save(parsed.records[0], False, "New label", "new")
    assert source.read_bytes() == original
    assert list(csv.DictReader(store.output_path.open(encoding="utf-8-sig")).fieldnames) == [
        "local_path",
        "BOOKMARKED",
        "Review_Label",
        "Tags",
    ]
    assert parse_results(store.output_path).records[0].values["review_label"] == "New label"


def test_atomic_annotation_save_failure_keeps_source_and_previous_copy(tmp_path, monkeypatch):
    from sscraper.csv_viewer import AnnotationStore

    source = write_csv(tmp_path / "results.csv", [{"local_path": "image.jpg"}])
    original = source.read_bytes()
    parsed = parse_results(source)
    store = AnnotationStore(source, parsed)
    store.save(parsed.records[0], True, "Saved", "old")
    previous_copy = store.output_path.read_bytes()

    def fail_replace(*_):
        raise OSError("Disk full")

    monkeypatch.setattr("sscraper.csv_viewer.os.replace", fail_replace)
    with pytest.raises(OSError, match="Disk full"):
        store.save(parsed.records[0], False, "Unsaved", "new")
    assert source.read_bytes() == original
    assert store.output_path.read_bytes() == previous_copy
    assert parsed.records[0].title == "Saved"
    assert parsed.records[0].bookmarked


def test_single_view_annotations_autosave_and_bookmark_shortcut_does_not_steal_typing(
    tmp_path, monkeypatch
):
    from PIL import Image
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from sscraper.csv_viewer import CSVViewer

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "image.png"
    Image.new("RGB", (200, 100), "blue").save(image)
    path = write_csv(
        tmp_path / "results.csv",
        [
            {"status": "downloaded", "local_path": str(image), "label": f"Image {index}"}
            for index in range(2)
        ],
    )
    original = path.read_bytes()
    viewer = CSVViewer()
    try:
        viewer.show()
        viewer.activateWindow()
        viewer.accept_csv(path, parse_results(path))
        app.processEvents()
        viewer.show_single(viewer.filtered[0])
        assert not list(tmp_path.glob("*.annotated*.csv"))
        QTest.keyClick(viewer.single_view, Qt.Key.Key_B)
        assert viewer.filtered[0].bookmarked
        viewer.review_label.setFocus()
        QTest.keyClicks(viewer.review_label, "Beach")
        assert viewer.filtered[0].bookmarked
        viewer.resize(850, 850)
        app.processEvents()
        assert viewer.review_label.text() == "Beach"
        viewer.review_tags.setFocus()
        QTest.keyClicks(viewer.review_tags, "travel, blue, Travel")
        QTest.keyClick(viewer.review_tags, Qt.Key.Key_Left)
        assert viewer.image_index == 0
        viewer.move_image(1)
        copy_path = viewer.annotation_store.output_path
        assert path.read_bytes() == original
        restored = parse_results(copy_path)
        assert restored.records[0].title == "Beach"
        assert restored.records[0].tags == "travel, blue"
        assert restored.records[0].bookmarked
        assert not restored.records[1].bookmarked
        viewer.show_cards()
        viewer.bookmarks_only.setChecked(True)
        assert len(viewer.filtered) == 1
        viewer.search.setText("blue")
        assert len(viewer.filtered) == 1
        viewer.search.clear()
        viewer.bookmark_card(viewer.filtered[0], False)
        assert not viewer.filtered
        assert not parse_results(copy_path).records[0].bookmarked
        assert path.read_bytes() == original
    finally:
        viewer.close()
