from __future__ import annotations

import csv
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImageReader, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

PAGE_SIZE = 24


@dataclass(frozen=True)
class Record:
    row_number: int
    values: dict[str, str]

    @property
    def title(self) -> str:
        return (
            self.values.get("review_label")
            or self.values.get("label")
            or self.values.get("relative_path")
            or self.filename
            or f"Row {self.row_number}"
        )

    @property
    def filename(self) -> str:
        raw = (
            self.values.get("local_path")
            or self.values.get("source_path")
            or self.values.get("copied_path", "")
        )
        return PureWindowsPath(raw).name if "\\" in raw else Path(raw).name if raw else ""

    @property
    def status(self) -> str:
        return self.values.get("status", "") or "saved"

    @property
    def bookmarked(self) -> bool:
        return self.values.get("bookmarked", "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @property
    def tags(self) -> str:
        return normalize_tags(self.values.get("tags", ""))

    @property
    def source_url(self) -> str:
        # The clickable watch/detail-page link takes priority over the listing URL.
        return next(
            (
                value
                for key in ("link_url", "page_url")
                if (value := self.values.get(key, "")) and is_web_url(value)
            ),
            "",
        )


@dataclass(frozen=True)
class ParsedCSV:
    records: list[Record]
    skipped_errors: int
    skipped_empty: int
    fieldnames: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)


def normalize_tags(value: str) -> str:
    tags = {}
    for tag in value.split(","):
        tag = tag.strip()
        if tag:
            tags.setdefault(tag.casefold(), tag)
    return ", ".join(tags.values())


class AnnotationStore:
    """Write review data only to a new CSV copy; retain every original row/column."""

    def __init__(self, source: Path, parsed: ParsedCSV):
        self.source = source.resolve()
        self.parsed = parsed
        self.output_path: Path | None = None
        self.headers = list(parsed.fieldnames)
        self.columns = {name.strip().lower(): name for name in self.headers}
        for name in ("bookmarked", "review_label", "tags"):
            if name not in self.columns:
                self.headers.append(name)
                self.columns[name] = name

    def write(self, rows):
        created = False
        target = self.output_path
        if target is None:
            suffix = 1
            while True:
                name = (
                    f"{self.source.stem}.annotated{('-' + str(suffix)) if suffix > 1 else ''}.csv"
                )
                target = self.source.with_name(name)
                try:
                    with target.open("x", encoding="utf-8"):
                        pass
                    created = True
                    break
                except FileExistsError:
                    suffix += 1
        if target.resolve() == self.source:
            raise ValueError("Annotations must be saved to a separate copy of the CSV.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", dir=target.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except (OSError, ValueError, csv.Error):
            if created:
                target.unlink(missing_ok=True)
            raise
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        self.output_path = target

    def save(self, record: Record, bookmarked: bool, label: str, tags: str):
        annotations = {
            "bookmarked": "true" if bookmarked else "false",
            "review_label": label.strip(),
            "tags": normalize_tags(tags),
        }
        index = record.row_number - 2
        if not 0 <= index < len(self.parsed.rows):
            raise ValueError("This row is no longer available. Reload the CSV.")
        updated = dict(self.parsed.rows[index])
        updated.update({self.columns[key]: value for key, value in annotations.items()})
        rows = list(self.parsed.rows)
        rows[index] = updated
        self.write(rows)
        self.parsed.rows[index].update(updated)
        record.values.update(annotations)


def is_web_url(value: str) -> bool:
    url = QUrl(value)
    return url.isValid() and url.scheme().lower() in {"http", "https"} and bool(url.host())


def parse_results(path: Path, interrupted=lambda: False) -> ParsedCSV:
    records, errors, empty = [], 0, 0
    original_rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The CSV is empty or has no header row.")
        original_headers = list(reader.fieldnames)
        headers = [header.strip().lower() for header in original_headers]
        if len(headers) != len(set(headers)):
            raise ValueError("The CSV contains duplicate column names.")
        if not {"local_path", "source_path", "copied_path", "image_url"}.intersection(headers):
            raise ValueError(
                "Choose a results CSV containing local_path, source_path, copied_path or image_url."
            )
        reader.fieldnames = headers
        for row_number, row in enumerate(reader, start=2):
            if interrupted():
                return ParsedCSV([], 0, 0)
            if None in row:
                raise ValueError(f"CSV row {row_number} has more values than column headers.")
            original_rows.append(
                {
                    original: row.get(normalized) or ""
                    for original, normalized in zip(original_headers, headers, strict=True)
                }
            )
            values = {key: (value or "").strip() for key, value in row.items()}
            if not any(values.values()):
                empty += 1
                continue
            status = values.get("status", "").lower()
            if values.get("error") or "error" in status or "fail" in status:
                errors += 1
                continue
            records.append(Record(row_number, values))
    return ParsedCSV(records, errors, empty, original_headers, original_rows)


def local_image(record: Record, csv_path: Path) -> Path | None:
    """Resolve original paths and recover images when an entire run folder was moved."""
    candidates = []
    for key in ("local_path", "copied_path", "source_path"):
        raw = record.values.get(key, "")
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend((csv_path.parent / path, Path.cwd() / path))
        name = PureWindowsPath(raw).name if "\\" in raw else path.name
        candidates.extend((csv_path.parent / "images" / name, csv_path.parent / name))
    relative = record.values.get("relative_path", "")
    if relative:
        candidates.append(csv_path.parent / "matches" / relative)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


class CSVLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            result = parse_results(self.path, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(result)
        except (OSError, ValueError, UnicodeError, csv.Error) as exc:
            self.failed.emit(str(exc))


class Thumbnail(QLabel):
    clicked = Signal()

    def __init__(self, height=158):
        super().__init__()
        self.original = QPixmap()
        if height is not None:
            self.setFixedHeight(height)
        else:
            self.setMinimumHeight(180)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed if height is not None else QSizePolicy.Policy.Ignored,
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("thumbnail")

    def show_pixmap(self, pixmap):
        self.original = pixmap
        self.fit()

    def fit(self):
        if not self.original.isNull():
            self.setPixmap(
                self.original.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class Card(QFrame):
    selected = Signal(object)
    bookmark_changed = Signal(object, bool)
    edit_requested = Signal(object)

    def __init__(self, record: Record, csv_path: Path):
        super().__init__()
        self.setObjectName("card")
        self.setFixedHeight(358)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        image = Thumbnail()
        image.setCursor(Qt.CursorShape.PointingHandCursor)
        image.setToolTip("Click to view this image; use Left/Right arrows to navigate")
        image.clicked.connect(lambda: self.selected.emit(record))
        path = local_image(record, csv_path)
        if path:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                size.scale(360, 158, Qt.AspectRatioMode.KeepAspectRatio)
                reader.setScaledSize(size)
            pixmap = QPixmap.fromImage(reader.read())
            if not pixmap.isNull():
                image.show_pixmap(pixmap)
            else:
                image.setText("Preview unavailable")
        else:
            image.setText("Local image not found")
        layout.addWidget(image)
        title = QLabel(record.title)
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        title.setFixedHeight(36)
        title.setToolTip(record.title)
        title.setObjectName("title")
        title_row = QHBoxLayout()
        title_row.addWidget(title, 1)
        bookmark = QPushButton("★" if record.bookmarked else "☆")
        bookmark.setToolTip("Toggle bookmark")
        bookmark.setCheckable(True)
        bookmark.setChecked(record.bookmarked)
        bookmark.setFixedWidth(38)
        bookmark.clicked.connect(lambda checked: self.bookmark_changed.emit(record, checked))
        title_row.addWidget(bookmark)
        edit = QPushButton("Edit…")
        edit.setToolTip("Edit label and tags")
        edit.clicked.connect(lambda: self.edit_requested.emit(record))
        title_row.addWidget(edit)
        layout.addLayout(title_row)
        detail = record.status.replace("_", " ")
        if record.values.get("distance"):
            detail += f" · distance {record.values['distance']}"
        status = QLabel(detail)
        status.setTextFormat(Qt.TextFormat.PlainText)
        status.setToolTip(detail)
        status.setStyleSheet(
            "color: #6de6b9;" if record.status == "candidate_match" else "color: #9baac0;"
        )
        layout.addWidget(status)
        tags = QLabel(record.tags or "No tags")
        tags.setTextFormat(Qt.TextFormat.PlainText)
        tags.setWordWrap(True)
        tags.setFixedHeight(30)
        tags.setToolTip(record.tags)
        tags.setStyleSheet("color: #9baac0;")
        layout.addWidget(tags)
        buttons = QHBoxLayout()
        open_image = QPushButton("Open image")
        remote_image = record.values.get("image_url", "")
        image_url = (
            QUrl.fromLocalFile(str(path))
            if path
            else QUrl(remote_image)
            if is_web_url(remote_image)
            else None
        )
        open_image.setEnabled(image_url is not None)
        open_image.setToolTip(str(path) if path else remote_image or "No image path available")
        open_image.clicked.connect(lambda: QDesktopServices.openUrl(image_url))
        buttons.addWidget(open_image)
        source = QPushButton("Open source")
        source.setEnabled(bool(record.source_url))
        source.setToolTip(record.source_url or "No source link in this row")
        source.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(record.source_url)))
        buttons.addWidget(source)
        layout.addLayout(buttons)


class CSVViewer(QMainWindow):
    def __init__(self, path: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SScraper · CSV thumbnail cards")
        self.resize(1180, 850)
        self.path = None
        self.parsed = ParsedCSV([], 0, 0)
        self.filtered = []
        self.page = 0
        self.image_index = 0
        self.single_mode = False
        self.annotation_store = None
        self.columns = 3
        self.loader = None
        self.close_after_loading = False
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        heading = QLabel("Thumbnail cards")
        heading.setStyleSheet("font-size: 26px; font-weight: 700;")
        layout.addWidget(heading)
        actions = QHBoxLayout()
        self.open_csv = QPushButton("Open results.csv…")
        self.open_csv.clicked.connect(self.choose_csv)
        actions.addWidget(self.open_csv)
        self.reload = QPushButton("Reload")
        self.reload.setEnabled(False)
        self.reload.clicked.connect(
            lambda: self.load(
                self.annotation_store.output_path
                if self.annotation_store and self.annotation_store.output_path
                else self.path
            )
        )
        actions.addWidget(self.reload)
        self.view_toggle = QPushButton("Single image")
        self.view_toggle.clicked.connect(self.toggle_view)
        actions.addWidget(self.view_toggle)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search title, file name, status or source URL")
        self.search.textChanged.connect(self.apply_filter)
        actions.addWidget(self.search, 1)
        self.status_filter = QComboBox()
        self.status_filter.addItem("All successful rows", "")
        self.status_filter.currentIndexChanged.connect(self.apply_filter)
        actions.addWidget(self.status_filter)
        self.bookmarks_only = QCheckBox("Bookmarked")
        self.bookmarks_only.toggled.connect(self.apply_filter)
        actions.addWidget(self.bookmarks_only)
        layout.addLayout(actions)
        self.description = QLabel(
            "Open a results CSV to view its saved thumbnails. Error rows are excluded."
        )
        self.description.setWordWrap(True)
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.description)
        self.annotation_status = QLabel(
            "Bookmarks, labels and tags are saved automatically to a new CSV copy."
        )
        self.annotation_status.setWordWrap(True)
        self.annotation_status.setTextFormat(Qt.TextFormat.PlainText)
        self.annotation_status.setStyleSheet("color: #9baac0;")
        layout.addWidget(self.annotation_status)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid.setSpacing(14)
        self.scroll.setWidget(self.container)
        self.views = QStackedWidget()
        self.views.addWidget(self.scroll)
        self.single_view = QWidget()
        self.single_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        single_layout = QVBoxLayout(self.single_view)
        single_actions = QHBoxLayout()
        self.single_title = QLabel()
        self.single_title.setWordWrap(True)
        self.single_title.setTextFormat(Qt.TextFormat.PlainText)
        self.single_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.single_title.setMaximumHeight(64)
        single_actions.addWidget(self.single_title, 1)
        self.single_source = QPushButton("Open source")
        self.single_source.clicked.connect(self.open_single_source)
        single_actions.addWidget(self.single_source)
        self.single_external = QPushButton("Open image")
        self.single_external.clicked.connect(self.open_single_external)
        single_actions.addWidget(self.single_external)
        single_layout.addLayout(single_actions)
        annotations = QHBoxLayout()
        self.single_bookmark = QPushButton("☆ Bookmark")
        self.single_bookmark.setCheckable(True)
        self.single_bookmark.clicked.connect(self.save_single_annotations)
        annotations.addWidget(self.single_bookmark)
        self.review_label = QLineEdit()
        self.review_label.setPlaceholderText("Your label")
        self.review_label.editingFinished.connect(self.save_single_annotations)
        annotations.addWidget(self.review_label)
        self.review_tags = QLineEdit()
        self.review_tags.setPlaceholderText("Tags, separated by commas")
        self.review_tags.editingFinished.connect(self.save_single_annotations)
        annotations.addWidget(self.review_tags, 1)
        self.save_review = QPushButton("Save")
        self.save_review.clicked.connect(self.save_single_annotations)
        annotations.addWidget(self.save_review)
        single_layout.addLayout(annotations)
        self.single_image = Thumbnail(height=None)
        self.single_image.clicked.connect(lambda: self.single_view.setFocus())
        single_layout.addWidget(self.single_image, 1)
        self.single_detail = QLabel("Left / Right: navigate · Escape: return to cards")
        self.single_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.single_detail.setWordWrap(True)
        single_layout.addWidget(self.single_detail)
        self.views.addWidget(self.single_view)
        layout.addWidget(self.views, 1)
        self.shortcuts = []
        self.typing_shortcuts = []
        for key, callback in [
            (Qt.Key.Key_Left, lambda: self.move_image(-1)),
            (Qt.Key.Key_Right, lambda: self.move_image(1)),
            (Qt.Key.Key_Escape, self.show_cards),
            (Qt.Key.Key_B, self.toggle_single_bookmark),
        ]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
            if key != Qt.Key.Key_Escape:
                self.typing_shortcuts.append(shortcut)
        navigation = QHBoxLayout()
        self.previous = QPushButton("Previous")
        self.previous.clicked.connect(lambda: self.change_page(-1))
        self.page_label = QLabel("No CSV loaded")
        self.next = QPushButton("Next")
        self.next.clicked.connect(lambda: self.change_page(1))
        navigation.addStretch()
        navigation.addWidget(self.previous)
        navigation.addWidget(self.page_label)
        navigation.addWidget(self.next)
        navigation.addStretch()
        layout.addLayout(navigation)
        QApplication.instance().focusChanged.connect(self.focus_changed)
        self.focus_changed(None, QApplication.focusWidget())
        self.render()
        if path:
            QTimer.singleShot(0, lambda: self.load(path))

    def choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open scan results",
            str(self.path.parent) if self.path else "",
            "CSV files (*.csv)",
        )
        if path:
            self.load(Path(path))

    def load(self, path: Path | None):
        if path is None or (self.loader and self.loader.isRunning()):
            return
        if not self.save_single_annotations():
            return
        candidate = Path(path).expanduser().resolve()
        self.close_after_loading = False
        self.open_csv.setEnabled(False)
        self.reload.setEnabled(False)
        self.description.setText(f"Reading {candidate}…")
        self.loader = CSVLoader(candidate, self)
        self.loader.loaded.connect(lambda result: self.accept_csv(candidate, result))
        self.loader.failed.connect(self.load_failed)
        self.loader.finished.connect(self.loading_finished)
        self.loader.start()

    def accept_csv(self, path, result):
        self.path, self.parsed = path, result
        self.annotation_store = AnnotationStore(path, result)
        self.annotation_status.setText(
            "Edits will be saved automatically to a new annotated CSV copy."
        )
        self.search.clear()
        self.status_filter.blockSignals(True)
        self.status_filter.clear()
        self.status_filter.addItem("All successful rows", "")
        for status in sorted({record.status for record in result.records}):
            self.status_filter.addItem(status.replace("_", " "), status)
        self.status_filter.blockSignals(False)
        self.description.setText(
            f"{path}\n{len(result.records)} cards · {result.skipped_errors} error rows excluded · {result.skipped_empty} empty rows skipped"
        )
        self.apply_filter()

    def load_failed(self, message):
        self.description.setText(f"Could not load CSV: {message}")
        QMessageBox.warning(self, "CSV could not be opened", message)

    def loading_finished(self):
        self.open_csv.setEnabled(True)
        self.reload.setEnabled(self.path is not None)
        if self.close_after_loading:
            QTimer.singleShot(0, self.close)

    def apply_filter(self, *_, reset_page=True):
        current = (
            self.filtered[self.image_index]
            if self.filtered and self.image_index < len(self.filtered)
            else None
        )
        text = self.search.text().strip().casefold()
        status = self.status_filter.currentData()
        self.filtered = [
            record
            for record in self.parsed.records
            if (not status or record.status == status)
            and (not self.bookmarks_only.isChecked() or record.bookmarked)
            and (not text or any(text in value.casefold() for value in record.values.values()))
        ]
        if reset_page:
            self.page = 0
        self.image_index = self.filtered.index(current) if current in self.filtered else 0
        self.render()

    def change_page(self, delta):
        if self.single_mode:
            self.move_image(delta)
            return
        self.page += delta
        self.render()
        self.scroll.verticalScrollBar().setValue(0)

    def toggle_view(self):
        if self.single_mode:
            self.show_cards()
        elif self.filtered:
            index = (
                self.image_index
                if self.image_index // PAGE_SIZE == self.page
                else self.page * PAGE_SIZE
            )
            self.show_single(self.filtered[min(index, len(self.filtered) - 1)])

    def show_single(self, record):
        if record not in self.filtered:
            return
        self.image_index = self.filtered.index(record)
        self.single_mode = True
        self.view_toggle.setText("Back to cards · Esc")
        self.views.setCurrentWidget(self.single_view)
        self.render_single()
        self.single_view.setFocus(Qt.FocusReason.OtherFocusReason)

    def show_cards(self):
        if not self.single_mode:
            return
        if not self.save_single_annotations():
            return
        self.single_mode = False
        self.page = self.image_index // PAGE_SIZE
        self.view_toggle.setText("Single image")
        self.views.setCurrentWidget(self.scroll)
        self.focus_changed(None, QApplication.focusWidget())
        self.apply_filter(reset_page=False)

    def move_image(self, delta):
        if self.single_mode and self.filtered:
            if not self.save_single_annotations():
                return
            self.image_index = max(0, min(self.image_index + delta, len(self.filtered) - 1))
            self.render_single()

    def render_single(self):
        self.single_image.original = QPixmap()
        self.single_image.clear()
        if not self.filtered:
            self.single_title.setText("No matching images")
            self.single_image.setText("Change your search or status filter")
            self.single_source.setEnabled(False)
            self.single_external.setEnabled(False)
            self.single_bookmark.setEnabled(False)
            self.review_label.setEnabled(False)
            self.review_tags.setEnabled(False)
            self.save_review.setEnabled(False)
            self.review_label.clear()
            self.review_tags.clear()
            self.previous.setEnabled(False)
            self.next.setEnabled(False)
            self.page_label.setText("0 images")
            self.single_detail.setText("Escape: return to cards")
            return
        self.image_index = max(0, min(self.image_index, len(self.filtered) - 1))
        record = self.filtered[self.image_index]
        self.single_bookmark.setEnabled(True)
        self.review_label.setEnabled(True)
        self.review_tags.setEnabled(True)
        self.save_review.setEnabled(True)
        self.single_bookmark.setChecked(record.bookmarked)
        self.single_bookmark.setText("★ Bookmarked" if record.bookmarked else "☆ Bookmark")
        self.review_label.setText(record.values.get("review_label", ""))
        self.review_tags.setText(record.tags)
        path = local_image(record, self.path)
        self.single_title.setText(record.title)
        self.single_title.setToolTip(record.title)
        if path:
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                size.scale(2048, 2048, Qt.AspectRatioMode.KeepAspectRatio)
                reader.setScaledSize(size)
            pixmap = QPixmap.fromImage(reader.read())
            if not pixmap.isNull():
                self.single_image.show_pixmap(pixmap)
            else:
                self.single_image.setText("Preview unavailable")
        else:
            self.single_image.setText("Local image not found")
        self.single_source.setEnabled(bool(record.source_url))
        self.single_source.setToolTip(record.source_url or "No source link in this row")
        self.single_external.setEnabled(
            bool(path) or is_web_url(record.values.get("image_url", ""))
        )
        self.single_external.setToolTip(str(path or record.values.get("image_url", "")))
        details = record.status.replace("_", " ")
        if record.values.get("distance"):
            details += f" · distance {record.values['distance']}"
        self.single_detail.setText(
            details + "\nLeft / Right: navigate · B: bookmark · Escape: return to cards"
        )
        self.previous.setEnabled(self.image_index > 0)
        self.next.setEnabled(self.image_index + 1 < len(self.filtered))
        self.page_label.setText(f"Image {self.image_index + 1} / {len(self.filtered)}")

    def open_single_source(self):
        if self.filtered:
            source = self.filtered[self.image_index].source_url
            if source:
                QDesktopServices.openUrl(QUrl(source))

    def focus_changed(self, _old, new):
        for shortcut in self.typing_shortcuts:
            shortcut.setEnabled(self.single_mode and not isinstance(new, (QLineEdit, QComboBox)))

    def save_annotation(self, record, bookmarked, label, tags):
        if not self.annotation_store:
            return False
        if (
            record.bookmarked == bookmarked
            and record.values.get("review_label", "") == label.strip()
            and record.tags == normalize_tags(tags)
        ):
            return True
        try:
            self.annotation_store.save(record, bookmarked, label, tags)
        except (OSError, ValueError, csv.Error) as exc:
            self.annotation_status.setText(f"Could not save annotations: {exc}")
            QMessageBox.warning(self, "Annotations were not saved", str(exc))
            return False
        self.annotation_status.setText(f"Saved to copy: {self.annotation_store.output_path}")
        return True

    def save_single_annotations(self, *_):
        if not self.single_mode or not self.filtered:
            return True
        record = self.filtered[self.image_index]
        if not self.save_annotation(
            record,
            self.single_bookmark.isChecked(),
            self.review_label.text(),
            self.review_tags.text(),
        ):
            return False
        self.single_bookmark.setText("★ Bookmarked" if record.bookmarked else "☆ Bookmark")
        self.single_title.setText(record.title)
        self.single_title.setToolTip(record.title)
        return True

    def toggle_single_bookmark(self):
        if self.single_mode and self.filtered:
            self.single_bookmark.click()

    def bookmark_card(self, record, checked):
        self.save_annotation(record, checked, record.values.get("review_label", ""), record.tags)
        self.apply_filter(reset_page=False)

    def edit_annotation(self, record):
        dialog = QDialog(self)
        dialog.setWindowTitle("Label and tags")
        dialog.resize(460, 230)
        form = QFormLayout(dialog)
        bookmark = QCheckBox("Bookmarked")
        bookmark.setChecked(record.bookmarked)
        label = QLineEdit(record.values.get("review_label", ""))
        tags = QLineEdit(record.tags)
        tags.setPlaceholderText("Separate tags with commas")
        form.addRow(bookmark)
        form.addRow("Your label", label)
        form.addRow("Tags", tags)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(dialog.reject)

        def save():
            if self.save_annotation(record, bookmark.isChecked(), label.text(), tags.text()):
                dialog.accept()

        buttons.accepted.connect(save)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_filter(reset_page=False)

    def open_single_external(self):
        if self.filtered:
            record = self.filtered[self.image_index]
            path = local_image(record, self.path)
            if path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            elif is_web_url(record.values.get("image_url", "")):
                QDesktopServices.openUrl(QUrl(record.values["image_url"]))

    def render(self):
        self.view_toggle.setEnabled(bool(self.filtered) or self.single_mode)
        if self.single_mode:
            self.render_single()
            return
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        pages = max(1, (len(self.filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page = max(0, min(self.page, pages - 1))
        start = self.page * PAGE_SIZE
        for index, record in enumerate(self.filtered[start : start + PAGE_SIZE]):
            card = Card(record, self.path)
            card.selected.connect(self.show_single)
            card.bookmark_changed.connect(self.bookmark_card)
            card.edit_requested.connect(self.edit_annotation)
            self.grid.addWidget(card, index // self.columns, index % self.columns)
        for column in range(4):
            self.grid.setColumnStretch(column, int(column < self.columns))
        if not self.filtered:
            message = QLabel(
                "No matching cards" if self.path else "Choose a results.csv file to begin"
            )
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(message, 0, 0, 1, self.columns)
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled(self.page + 1 < pages)
        self.page_label.setText(f"Page {self.page + 1} / {pages} · {len(self.filtered)} cards")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, min(4, (self.width() - 70) // 320))
        if columns != self.columns:
            self.columns = columns
            if not self.single_mode:
                self.render()

    def closeEvent(self, event):
        if not self.save_single_annotations():
            event.ignore()
            return
        if self.loader and self.loader.isRunning():
            self.close_after_loading = True
            self.loader.requestInterruption()
            event.ignore()
        else:
            event.accept()


STYLE = """
QMainWindow, QWidget { background: #151d2a; color: #e0e7f1; font-size: 12px; }
QFrame#card { background: #1c2839; border: 1px solid #34425a; border-radius: 8px; }
QFrame#card QLabel { background: transparent; border: none; }
QLabel#thumbnail { background: #101722; border-radius: 5px; }
QLabel#title { font-weight: 600; }
QPushButton { background: #26364b; border: 1px solid #3b4d68; border-radius: 5px; padding: 8px 12px; }
QPushButton:hover { background: #354b64; }
QPushButton:disabled { color: #66758a; }
QLineEdit, QComboBox { background: #101722; border: 1px solid #34425a; border-radius: 5px; padding: 8px; }
QScrollArea { border: none; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SScraper CSV viewer")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = CSVViewer(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
