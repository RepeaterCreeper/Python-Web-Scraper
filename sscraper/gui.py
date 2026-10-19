from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .pipeline import ScanConfig, ScanPipeline
from .scraping import validate_url


class ScanWorker(QThread):
    update = Signal(str, object)

    def __init__(self, config: ScanConfig):
        super().__init__()
        self.config = config
        self.stop = threading.Event()

    def run(self):
        ScanPipeline(self.config, self.update.emit, self.stop).run()


class Preview(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)
        self.image = QLabel("Waiting for images")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(180, 150)
        self.image.setMaximumHeight(250)
        self.image.setStyleSheet("background: #101722; border-radius: 6px; color: #8290a6;")
        self.status = QLabel("Idle")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(42)
        self.status.setMaximumHeight(64)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.status)
        self.pixmap = QPixmap()

    def show_image(self, data: dict):
        pixmap = QPixmap(data["path"])
        if not pixmap.isNull():
            painter = QPainter(pixmap)
            color = QColor("#5de4b4" if data.get("matched") else "#e4b95d")
            painter.setPen(QPen(color, max(2, pixmap.width() // 180)))
            for area in data.get("areas", []):
                painter.drawRect(int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"]))
            painter.end()
        self.pixmap = pixmap
        self.fit()

    def fit(self):
        if not self.pixmap.isNull():
            self.image.setPixmap(
                self.pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def reset(self):
        self.pixmap = QPixmap()
        self.image.clear()
        self.image.setText("Waiting for images")
        self.status.setText("Idle")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SScraper · Image & face scanner")
        self.resize(1260, 850)
        self.settings = QSettings("SScraper", "Desktop")
        self.worker: ScanWorker | None = None
        self.run_dir: Path | None = None
        self.close_when_stopped = False
        self.results: list[dict] = []
        self.csv_viewer = None
        self.setup_ui()
        self.restore_settings()

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        title = QLabel("SScraper")
        title.setStyleSheet("font-size: 27px; font-weight: 700; color: #f3f6fa;")
        subtitle = QLabel("Live image downloads · optional face matching · pages and feeds")
        subtitle.setStyleSheet("color: #9baac0; font-size: 13px;")
        outer.addWidget(title)
        outer.addWidget(subtitle)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        self.setup = QWidget()
        setup_layout = QVBoxLayout(self.setup)
        setup_layout.setContentsMargins(0, 8, 8, 0)
        form_group = QGroupBox("01 · Setup")
        form = QFormLayout(form_group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.url = QLineEdit()
        self.url.setPlaceholderText("https://example.com/gallery")
        form.addRow("Page URL", self.url)
        self.page_type = QComboBox()
        self.page_type.addItem("Single page", "single")
        self.page_type.addItem("Infinite scroll", "infinite")
        self.page_type.addItem("Pagination / Next page", "pagination")
        form.addRow("How does this site load more images?", self.page_type)
        self.references = QListWidget()
        self.references.setIconSize(QSize(46, 46))
        self.references.setMaximumHeight(160)
        self.references.setMinimumHeight(110)
        reference_label = QLabel("Reference photos · optional; one face per photo")
        reference_label.setWordWrap(True)
        form.addRow(reference_label, self.references)
        reference_buttons = QHBoxLayout()
        add = QPushButton("Add photos…")
        remove = QPushButton("Remove selected")
        add.clicked.connect(self.add_references)
        remove.clicked.connect(self.remove_reference)
        reference_buttons.addWidget(add)
        reference_buttons.addWidget(remove)
        form.addRow(reference_buttons)
        download_note = QLabel(
            "Leave photos empty to download images only. Add photos to enable DeepFace matching."
        )
        download_note.setWordWrap(True)
        download_note.setStyleSheet("color: #9baac0; font-size: 11px;")
        form.addRow(download_note)
        self.model = QComboBox()
        self.model.addItems(["SFace", "Facenet512", "ArcFace", "VGG-Face"])
        form.addRow("Face model", self.model)
        self.detector = QComboBox()
        self.detector.addItems(["opencv", "retinaface"])
        form.addRow("Face detector", self.detector)
        self.browser = QCheckBox("Render JavaScript (required for infinite scroll)")
        form.addRow(self.browser)
        self.output = QLineEdit(str(Path.cwd() / "runs"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        browse = QPushButton("…")
        browse.setMaximumWidth(36)
        browse.clicked.connect(self.choose_output)
        output_row.addWidget(browse)
        form.addRow("Save scans in", output_row)
        setup_layout.addWidget(form_group)

        advanced = QGroupBox("02 · Optional settings")
        advanced_form = QFormLayout(advanced)
        advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.selector = QLineEdit()
        self.selector.setPlaceholderText("Blank = all images; e.g. .gallery img")
        advanced_form.addRow("Image / container CSS selector", self.selector)
        self.limit = QSpinBox()
        self.limit.setRange(0, 1_000_000)
        self.limit.setSpecialValueText("All discovered images")
        advanced_form.addRow("Image limit", self.limit)
        self.delay = QDoubleSpinBox()
        self.delay.setRange(0, 60)
        self.delay.setSingleStep(0.25)
        self.delay.setValue(0.25)
        self.delay.setSuffix(" seconds")
        advanced_form.addRow("Delay between downloads", self.delay)
        self.next_selector = QLineEdit()
        self.next_selector.setPlaceholderText("Blank = detect Next; e.g. a.next or #load-next")
        advanced_form.addRow("Next link / button CSS selector", self.next_selector)
        self.page_param = QLineEdit()
        self.page_param.setPlaceholderText("Blank = follow Next; e.g. page, p, or offset")
        advanced_form.addRow("Pagination URL parameter", self.page_param)
        self.extra_params = QLineEdit()
        self.extra_params.setPlaceholderText("e.g. sort=newest&limit=20")
        advanced_form.addRow("Extra query parameters", self.extra_params)
        self.page_start = QSpinBox()
        self.page_start.setRange(0, 1_000_000_000)
        self.page_start.setValue(1)
        advanced_form.addRow("Starting page / offset", self.page_start)
        self.page_step = QSpinBox()
        self.page_step.setRange(1, 1_000_000_000)
        self.page_step.setValue(1)
        advanced_form.addRow("Page / offset increment", self.page_step)
        self.page_workers = QSpinBox()
        self.page_workers.setRange(1, 16)
        self.page_workers.setValue(4)
        self.page_workers.setToolTip(
            "Fetch this many parameter-based pages concurrently. Uses separate browsers when rendering JavaScript."
        )
        advanced_form.addRow("Parallel page workers", self.page_workers)
        self.scroll_selector = QLineEdit()
        self.scroll_selector.setPlaceholderText("Leave blank to scroll the whole page")
        self.scroll_selector.setToolTip(
            "Only set this if images load inside a scrolling panel. "
            "Enter a CSS selector that exists on the page; missing selectors fall back to whole-page scrolling."
        )
        advanced_form.addRow("Scroll container CSS selector · optional", self.scroll_selector)
        self.max_pages = QSpinBox()
        self.max_pages.setRange(1, 100_000)
        self.max_pages.setValue(100)
        advanced_form.addRow("Maximum pages", self.max_pages)
        self.max_scrolls = QSpinBox()
        self.max_scrolls.setRange(1, 100_000)
        self.max_scrolls.setValue(500)
        self.max_scrolls.setToolTip(
            "Consecutive attempts without moving forward or discovering new images. "
            "Resets to zero whenever scrolling succeeds."
        )
        advanced_form.addRow("Maximum scroll attempts without progress", self.max_scrolls)
        self.idle_rounds = QSpinBox()
        self.idle_rounds.setRange(1, 100)
        self.idle_rounds.setValue(3)
        advanced_form.addRow("Stop after attempts with no new images", self.idle_rounds)
        self.wait_seconds = QDoubleSpinBox()
        self.wait_seconds.setRange(0.1, 60)
        self.wait_seconds.setValue(1.5)
        self.wait_seconds.setSuffix(" seconds")
        advanced_form.addRow("Wait for new content / between pages", self.wait_seconds)
        self.max_minutes = QSpinBox()
        self.max_minutes.setRange(0, 10_000)
        self.max_minutes.setValue(30)
        self.max_minutes.setSpecialValueText("No time limit")
        self.max_minutes.setSuffix(" minutes")
        advanced_form.addRow("Discovery time limit", self.max_minutes)
        threshold_row = QHBoxLayout()
        self.custom_threshold = QCheckBox("Custom")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 2)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.593)
        self.threshold.setEnabled(False)
        self.custom_threshold.toggled.connect(self.update_reference_mode)
        threshold_row.addWidget(self.custom_threshold)
        threshold_row.addWidget(self.threshold)
        advanced_form.addRow("Cosine-distance threshold", threshold_row)
        note = QLabel(
            "Default uses the model’s threshold. Lower distances mean closer matches. "
            "A match to any reference is a candidate for manual review."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9baac0; font-size: 11px;")
        advanced_form.addRow(note)
        setup_layout.addWidget(advanced)
        setup_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.setup)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(430)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        split.addWidget(scroll)

        workspace = QWidget()
        work = QVBoxLayout(workspace)
        work.setContentsMargins(10, 8, 0, 0)
        actions = QHBoxLayout()
        self.start = QPushButton("Start scan")
        self.start.setObjectName("start")
        self.start.clicked.connect(self.start_scan)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_scan)
        self.open_folder = QPushButton("Open scan folder")
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(self.show_output)
        actions.addWidget(self.start)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        actions.addWidget(self.open_folder)
        work.addLayout(actions)
        display_options = QHBoxLayout()
        self.show_previews = QCheckBox("Show previews")
        self.show_previews.toggled.connect(self.update_previews)
        display_options.addWidget(self.show_previews)
        self.hide_url = QCheckBox("Hide URL")
        self.hide_url.setToolTip("Mask the Page URL input with password dots.")
        self.hide_url.toggled.connect(self.update_url_visibility)
        display_options.addWidget(self.hide_url)
        display_options.addStretch()
        work.addLayout(display_options)
        self.counts = QLabel("Found 0  ·  Downloaded 0  ·  Processed 0  ·  Matches 0  ·  Errors 0")
        self.counts.setStyleSheet("padding: 8px 0; color: #b7c5d8;")
        work.addWidget(self.counts)
        self.crawl_info = QLabel("Single page · ready")
        self.crawl_info.setWordWrap(True)
        self.crawl_info.setStyleSheet("color: #9baac0;")
        work.addWidget(self.crawl_info)
        previews = QHBoxLayout()
        self.scan_preview = Preview("Scanning · image download")
        self.process_preview = Preview("Processing · face comparison")
        previews.addWidget(self.scan_preview)
        previews.addWidget(self.process_preview)
        work.addLayout(previews)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        work.addWidget(self.progress)

        result_actions = QHBoxLayout()
        result_actions.addWidget(QLabel("Results"))
        csv_cards = QPushButton("View CSV cards…")
        csv_cards.clicked.connect(self.show_csv_cards)
        result_actions.addWidget(csv_cards)
        result_actions.addStretch()
        self.matches_only = QCheckBox("Candidates only")
        self.matches_only.toggled.connect(self.filter_results)
        result_actions.addWidget(self.matches_only)
        open_source = QPushButton("Open selected source")
        open_source.clicked.connect(self.show_source)
        result_actions.addWidget(open_source)
        work.addLayout(result_actions)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Image", "Result", "Distance", "Threshold", "Refs", "Faces", "Source"]
        )
        self.table.verticalHeader().hide()
        self.table.setIconSize(QSize(55, 40))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self.show_local_image)
        work.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(100)
        self.log.setPlaceholderText("Live activity and errors will appear here.")
        work.addWidget(self.log)
        split.addWidget(workspace)
        split.setSizes([360, 850])
        self.statusBar().showMessage(
            "Enter a non-explicit page URL. Reference photos are optional."
        )
        self.references.model().rowsInserted.connect(self.update_reference_mode)
        self.references.model().rowsRemoved.connect(self.update_reference_mode)
        self.page_type.currentIndexChanged.connect(self.update_page_type)
        self.browser.toggled.connect(self.update_page_type)
        self.page_param.textChanged.connect(self.update_page_type)
        self.update_reference_mode()
        self.update_page_type()
        self.update_previews()

    def update_url_visibility(self, hidden: bool):
        self.url.setEchoMode(
            QLineEdit.EchoMode.Password if hidden else QLineEdit.EchoMode.Normal
        )

    def update_previews(self):
        visible = self.show_previews.isChecked()
        for preview in (self.scan_preview, self.process_preview):
            preview.setVisible(visible)
            if not visible:
                preview.pixmap = QPixmap()
                preview.image.clear()
                preview.image.setText("Waiting for images")
        for row, data in enumerate(self.results):
            self.table.item(row, 0).setIcon(QIcon(data["local_path"]) if visible else QIcon())

    def update_page_type(self, *_):
        mode = self.page_type.currentData()
        infinite = mode == "infinite"
        if infinite:
            self.browser.setChecked(True)
        self.browser.setEnabled(not infinite)
        parameter_pages = mode == "pagination" and bool(self.page_param.text().strip())
        self.page_param.setEnabled(mode == "pagination")
        for field in (self.page_start, self.page_step, self.page_workers, self.extra_params):
            field.setEnabled(parameter_pages)
        self.next_selector.setEnabled(mode == "pagination" and not parameter_pages)
        self.max_pages.setEnabled(mode == "pagination")
        self.max_scrolls.setEnabled(infinite)
        self.idle_rounds.setEnabled(infinite)
        self.scroll_selector.setEnabled(infinite or self.browser.isChecked())

    def update_reference_mode(self, *_):
        has_references = self.references.count() > 0
        self.model.setEnabled(has_references)
        self.detector.setEnabled(has_references)
        self.custom_threshold.setEnabled(has_references)
        self.threshold.setEnabled(has_references and self.custom_threshold.isChecked())
        self.matches_only.setEnabled(has_references)
        for column in (2, 3, 4, 5):
            self.table.setColumnHidden(column, not has_references)
        if not has_references:
            self.matches_only.setChecked(False)

    def add_references(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose reference photos", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        )
        existing = {
            self.references.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.references.count())
        }
        for path in paths:
            if path not in existing:
                item = QListWidgetItem(QIcon(path), Path(path).name)
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                self.references.addItem(item)
                existing.add(path)

    def remove_reference(self):
        self.references.takeItem(self.references.currentRow())

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Scan output folder", self.output.text())
        if directory:
            self.output.setText(directory)

    def restore_settings(self):
        self.show_previews.setChecked(self.settings.value("show_previews", False, type=bool))
        self.hide_url.setChecked(self.settings.value("hide_url", False, type=bool))
        for key, field in [
            ("url", self.url),
            ("output", self.output),
            ("selector", self.selector),
            ("next_selector", self.next_selector),
            ("page_param", self.page_param),
            ("extra_params", self.extra_params),
            ("scroll_selector", self.scroll_selector),
        ]:
            field.setText(str(self.settings.value(key, field.text())))
        self.model.setCurrentText(str(self.settings.value("model", "SFace")))
        self.detector.setCurrentText(str(self.settings.value("detector", "opencv")))
        self.browser.setChecked(self.settings.value("browser", False, type=bool))
        index = self.page_type.findData(str(self.settings.value("page_type", "single")))
        self.page_type.setCurrentIndex(max(0, index))
        for key, field in [
            ("max_pages", self.max_pages),
            ("page_start", self.page_start),
            ("page_step", self.page_step),
            ("page_workers", self.page_workers),
            ("max_scrolls", self.max_scrolls),
            ("idle_rounds", self.idle_rounds),
            ("max_minutes", self.max_minutes),
        ]:
            field.setValue(self.settings.value(key, field.value(), type=int))
        self.wait_seconds.setValue(self.settings.value("wait_seconds", 1.5, type=float))
        self.update_page_type()

    def save_settings(self):
        self.settings.setValue("show_previews", self.show_previews.isChecked())
        self.settings.setValue("hide_url", self.hide_url.isChecked())
        for key, field in [
            ("url", self.url),
            ("output", self.output),
            ("selector", self.selector),
            ("next_selector", self.next_selector),
            ("page_param", self.page_param),
            ("extra_params", self.extra_params),
            ("scroll_selector", self.scroll_selector),
        ]:
            self.settings.setValue(key, field.text())
        self.settings.setValue("model", self.model.currentText())
        self.settings.setValue("detector", self.detector.currentText())
        self.settings.setValue("browser", self.browser.isChecked())
        self.settings.setValue("page_type", self.page_type.currentData())
        for key, field in [
            ("max_pages", self.max_pages),
            ("page_start", self.page_start),
            ("page_step", self.page_step),
            ("page_workers", self.page_workers),
            ("max_scrolls", self.max_scrolls),
            ("idle_rounds", self.idle_rounds),
            ("max_minutes", self.max_minutes),
            ("wait_seconds", self.wait_seconds),
        ]:
            self.settings.setValue(key, field.value())

    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return
        try:
            url = validate_url(self.url.text())
            references = [
                Path(self.references.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.references.count())
            ]
            if references and sys.version_info >= (3, 14):
                raise ValueError(
                    "Face matching requires Python 3.11-3.13. Remove reference photos "
                    "to download images on this Python version, or use Python 3.13."
                )
            if any(not path.is_file() for path in references):
                raise ValueError(
                    "A reference photo no longer exists. Remove it and choose another."
                )
            if not self.output.text().strip():
                raise ValueError("Choose an output folder.")
            # Validate selector before loading recognition models.
            from bs4 import BeautifulSoup

            for field in (self.selector, self.next_selector, self.scroll_selector):
                if field.isEnabled() and field.text().strip():
                    BeautifulSoup("", "html.parser").select(field.text())
            config = ScanConfig(
                url=url,
                references=references,
                output=Path(self.output.text()).expanduser().resolve(),
                model=self.model.currentText(),
                detector=self.detector.currentText(),
                threshold=self.threshold.value() if self.custom_threshold.isChecked() else None,
                selector=self.selector.text().strip(),
                browser=self.browser.isChecked(),
                delay=self.delay.value(),
                max_images=self.limit.value(),
                page_type=self.page_type.currentData(),
                next_selector=self.next_selector.text().strip(),
                max_pages=self.max_pages.value(),
                page_param=self.page_param.text().strip(),
                page_start=self.page_start.value(),
                page_step=self.page_step.value(),
                page_workers=self.page_workers.value(),
                extra_params=self.extra_params.text().strip(),
                max_scrolls=self.max_scrolls.value(),
                idle_rounds=self.idle_rounds.value(),
                wait_seconds=self.wait_seconds.value(),
                max_seconds=self.max_minutes.value() * 60,
                scroll_selector=self.scroll_selector.text().strip()
                if self.scroll_selector.isEnabled()
                else "",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Check setup", str(exc))
            return
        self.save_settings()
        self.results.clear()
        self.table.setRowCount(0)
        self.log.clear()
        self.run_dir = None
        self.open_folder.setEnabled(False)
        self.scan_preview.reset()
        self.process_preview.reset()
        self.process_preview.setTitle(
            "Processing · face comparison" if references else "Saved locally · download only"
        )
        self.counts.setText("Found 0  ·  Downloaded 0  ·  Processed 0  ·  Matches 0  ·  Errors 0")
        self.progress.setRange(0, 0)
        self.crawl_info.setText(f"{self.page_type.currentText()} · starting…")
        self.progress.setFormat("Scanning…")
        self.setup.setEnabled(False)
        self.start.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = ScanWorker(config)
        self.worker.update.connect(self.handle_update)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def stop_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop.set()
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage(
                "Stopping after the current download or model operation finishes…"
            )

    def handle_update(self, name: str, data):
        if name == "scan_status":
            self.scan_preview.status.setText(str(data))
        elif name == "crawl_progress":
            self.crawl_info.setText(
                f"Pages {data['pages']} · Attempts without progress {data.get('scroll_attempts', 0)} · Total scrolls {data['scrolls']} · "
                f"{data['new_images']} new images in this batch"
            )
        elif name == "process_status":
            self.process_preview.status.setText(str(data))
        elif name == "scan_image":
            if self.show_previews.isChecked():
                self.scan_preview.show_image(data)
        elif name == "process_image":
            if self.show_previews.isChecked():
                self.process_preview.show_image(data)
        elif name == "stats":
            self.counts.setText(
                "  ·  ".join(
                    f"{key.title()} {data[key]}"
                    for key in ["found", "downloaded", "processed", "matches", "errors"]
                )
            )
        elif name == "result":
            self.add_result(data)
        elif name == "run_dir":
            self.run_dir = Path(data)
            self.open_folder.setEnabled(True)
            self.log.appendPlainText(f"Saving results to {data}")
        elif name in {"log", "error"}:
            self.log.appendPlainText(("ERROR: " if name == "error" else "") + str(data))
            if name == "error":
                self.statusBar().showMessage(str(data))
        elif name == "done":
            outcome = data["outcome"].replace("_", " ").title()
            reason = data["discovery_reason"].replace("_", " ") or outcome.lower()
            self.crawl_info.setText(
                f"Pages {data['pages']} · Attempts without progress {data.get('scroll_attempts', 0)} · Total scrolls {data['scrolls']} · Stopped: {reason}"
            )
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.progress.setFormat(outcome)
            self.scan_preview.status.setText(outcome)
            download_only = data["mode"] == "download_only"
            self.process_preview.status.setText(
                f"{outcome} · {data['downloaded']} images saved"
                if download_only
                else f"{outcome} · {data['processed']} images processed"
            )
            detail = (
                f"{data['downloaded']} images saved"
                if download_only
                else f"{data['matches']} candidate matches"
            )
            self.statusBar().showMessage(f"{outcome}. {detail}. Results: {data['run_dir']}")
            self.log.appendPlainText(
                f"{outcome}. {data['unprocessed']} downloaded images left unprocessed."
            )

    def worker_finished(self):
        self.setup.setEnabled(True)
        self.update_page_type()
        self.update_reference_mode()
        self.start.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.close_when_stopped:
            QTimer.singleShot(0, self.close)

    def add_result(self, data: dict):
        index = self.table.rowCount()
        self.results.append(data)
        self.table.insertRow(index)
        values = [
            data.get("label") or Path(data["local_path"]).name[:10],
            data["status"].replace("_", " "),
            f"{data['distance']:.4f}" if "distance" in data else "—",
            f"{data['threshold']:.4f}" if "threshold" in data else "—",
            str(data.get("supporting_references", "—")),
            str(data.get("face_count", "—")),
            data["image_url"],
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0 and self.show_previews.isChecked():
                item.setIcon(QIcon(data["local_path"]))
            item.setToolTip(data.get("error") or value)
            if data["status"] == "candidate_match":
                item.setForeground(QColor("#5de4b4"))
            self.table.setItem(index, column, item)
        self.table.setRowHeight(index, 48)
        self.table.setRowHidden(
            index, self.matches_only.isChecked() and data["status"] != "candidate_match"
        )

    def filter_results(self):
        for index, data in enumerate(self.results):
            self.table.setRowHidden(
                index, self.matches_only.isChecked() and data["status"] != "candidate_match"
            )

    def show_source(self):
        index = self.table.currentRow()
        if index >= 0:
            url = self.results[index]["link_url"]
            if QUrl(url).scheme() in {"http", "https"}:
                QDesktopServices.openUrl(QUrl(url))

    def show_csv_cards(self):
        from .csv_viewer import STYLE as CSV_STYLE
        from .csv_viewer import CSVViewer

        if self.csv_viewer is None:
            self.csv_viewer = CSVViewer()
            self.csv_viewer.setStyleSheet(CSV_STYLE)
        self.csv_viewer.show()
        self.csv_viewer.raise_()
        self.csv_viewer.activateWindow()
        report = self.run_dir / "results.csv" if self.run_dir else None
        if report and report.is_file():
            self.csv_viewer.load(report)
        else:
            self.csv_viewer.choose_csv()

    def show_local_image(self, row: int, _column: int):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.results[row]["local_path"]))

    def show_output(self):
        if self.run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.close_when_stopped = True
            self.stop_scan()
            event.ignore()
        else:
            self.save_settings()
            event.accept()


STYLE = """
QMainWindow, QWidget { background: #151d2a; color: #e0e7f1; font-size: 12px; }
QGroupBox { border: 1px solid #324054; border-radius: 8px; margin-top: 13px; padding: 12px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #b5c6df; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QPlainTextEdit, QTableWidget {
    background: #101722; border: 1px solid #34425a; border-radius: 5px; padding: 6px;
    selection-background-color: #304964;
}
QPushButton { background: #26364b; border: 1px solid #3b4d68; border-radius: 5px; padding: 8px 10px; }
QPushButton:hover { background: #354b64; }
QPushButton:disabled, QWidget:disabled { color: #66758a; }
QPushButton#start { background: #6de6b9; color: #10251c; font-weight: 700; border: none; }
QPushButton#start:disabled { background: #304f47; color: #79978b; }
QHeaderView::section { background: #202c3e; border: none; padding: 7px; }
QTableWidget { gridline-color: #28354a; }
QProgressBar { border: none; border-radius: 4px; background: #253247; text-align: center; height: 18px; }
QProgressBar::chunk { background: #3b715e; border-radius: 4px; }
QSplitter::handle { background: #253247; width: 1px; }
QScrollArea { border: none; }
QStatusBar { color: #9eafc6; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SScraper")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
