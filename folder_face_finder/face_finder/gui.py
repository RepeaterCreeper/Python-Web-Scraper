from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImageReader, QPainter, QPen, QPixmap
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .engine import Config, FolderScan


class ScanWorker(QThread):
    update = Signal(str, object)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stop = threading.Event()

    def run(self):
        FolderScan(self.config, self.update.emit, self.stop).run()


class Preview(QGroupBox):
    def __init__(self):
        super().__init__("Current image")
        layout = QVBoxLayout(self)
        self.image = QLabel("Choose a folder to begin")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(240, 180)
        self.image.setMaximumHeight(300)
        self.image.setStyleSheet("background: #101722; border-radius: 6px; color: #8290a6;")
        self.caption = QLabel("Idle")
        self.caption.setWordWrap(True)
        self.caption.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.caption)
        self.pixmap = QPixmap()

    def show_image(self, data):
        reader = QImageReader(data["path"])
        reader.setAutoTransform(True)
        pixmap = QPixmap.fromImage(reader.read())
        if not pixmap.isNull():
            painter = QPainter(pixmap)
            for index, area in enumerate(data.get("areas", [])):
                is_match = data.get("matched") and index == data.get("closest_face")
                painter.setPen(
                    QPen(
                        QColor("#5de4b4" if is_match else "#e4b95d"), max(2, pixmap.width() // 180)
                    )
                )
                painter.drawRect(int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"]))
            painter.end()
        self.pixmap = pixmap
        self.caption.setText(Path(data["path"]).name)
        self.fit()
        if pixmap.isNull():
            self.image.setText("Preview unavailable")

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Folder Face Finder · DeepFace")
        self.resize(1200, 850)
        self.settings = QSettings("FolderFaceFinder", "Desktop")
        self.worker = None
        self.run_dir = None
        self.results = []
        self.close_after_stop = False
        self.build_ui()
        self.restore_settings()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        title = QLabel("Folder Face Finder")
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #f3f6fa;")
        outer.addWidget(title)
        outer.addWidget(QLabel("Search your local image folders with DeepFace"))
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)
        self.setup = QWidget()
        setup_layout = QVBoxLayout(self.setup)
        setup_layout.setContentsMargins(0, 8, 10, 0)
        group = QGroupBox("Scan setup")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.source = QLineEdit()
        self.source.setPlaceholderText("Choose a folder of images")
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_browse = QPushButton("Browse…")
        source_browse.clicked.connect(lambda: self.choose_folder(self.source))
        source_row.addWidget(source_browse)
        form.addRow("Image folder", source_row)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        form.addRow(self.recursive)
        self.references = QListWidget()
        self.references.setIconSize(QSize(42, 42))
        self.references.setFixedHeight(120)
        form.addRow("Reference photos · optional", self.references)
        buttons = QHBoxLayout()
        add = QPushButton("Add photos…")
        add.clicked.connect(self.add_references)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(lambda: self.references.takeItem(self.references.currentRow()))
        buttons.addWidget(add)
        buttons.addWidget(remove)
        form.addRow(buttons)
        explanation = QLabel(
            "Add photos of the same person to find matches. Each reference must contain one face. "
            "Without references, detect faces only."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #9baac0; font-size: 11px;")
        form.addRow(explanation)
        self.mode_label = QLabel("Mode: face detection only")
        self.mode_label.setWordWrap(True)
        form.addRow(self.mode_label)
        self.model = QComboBox()
        self.model.addItems(["SFace", "Facenet512", "ArcFace", "VGG-Face"])
        form.addRow("Recognition model", self.model)
        self.detector = QComboBox()
        self.detector.addItems(["opencv", "retinaface"])
        form.addRow("Face detector", self.detector)
        threshold_row = QHBoxLayout()
        self.custom_threshold = QCheckBox("Custom")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 2)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.593)
        self.custom_threshold.toggled.connect(self.update_mode)
        threshold_row.addWidget(self.custom_threshold)
        threshold_row.addWidget(self.threshold)
        form.addRow("Cosine-distance threshold", threshold_row)
        self.output = QLineEdit(str(Path(__file__).resolve().parents[1] / "runs"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        output_browse = QPushButton("…")
        output_browse.setMaximumWidth(36)
        output_browse.clicked.connect(lambda: self.choose_folder(self.output))
        output_row.addWidget(output_browse)
        form.addRow("Save reports in", output_row)
        self.copy_matches = QCheckBox("Copy candidate matches into the results folder")
        self.copy_matches.setChecked(True)
        form.addRow(self.copy_matches)
        note = QLabel("Lower distance means a closer match. Candidate matches need manual review.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9baac0; font-size: 11px;")
        form.addRow(note)
        setup_layout.addWidget(group)
        setup_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.setup)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(420)
        split.addWidget(scroll)

        workspace = QWidget()
        work = QVBoxLayout(workspace)
        work.setContentsMargins(10, 8, 0, 0)
        actions = QHBoxLayout()
        self.start = QPushButton("Start scan")
        self.start.setObjectName("start")
        self.start.clicked.connect(self.start_scan)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        self.open_output = QPushButton("Open results folder")
        self.open_output.setEnabled(False)
        self.open_output.clicked.connect(self.show_output)
        actions.addWidget(self.start)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        actions.addWidget(self.open_output)
        work.addLayout(actions)
        self.counts = QLabel("Found 0 · Processed 0 · Matches 0 · Errors 0")
        work.addWidget(self.counts)
        self.preview = Preview()
        work.addWidget(self.preview)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setFormat("Ready")
        self.progress.setValue(0)
        work.addWidget(self.progress)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Results · double-click to open an image"))
        filter_row.addStretch()
        self.filter = QComboBox()
        self.filter.addItem("All images", "all")
        self.filter.addItem("Candidates", "candidate_match")
        self.filter.addItem("Images with faces", "with_faces")
        self.filter.addItem("No faces", "no_face")
        self.filter.addItem("Errors", "processing_error")
        self.filter.currentIndexChanged.connect(self.filter_results)
        filter_row.addWidget(self.filter)
        work.addLayout(filter_row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Image", "Result", "Faces", "Distance", "Threshold", "Refs"]
        )
        self.table.setIconSize(QSize(44, 38))
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(
            lambda row, _: QDesktopServices.openUrl(
                QUrl.fromLocalFile(self.results[row]["source_path"])
            )
        )
        work.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(95)
        self.log.setPlaceholderText("Scan activity and errors")
        work.addWidget(self.log)
        split.addWidget(workspace)
        split.setSizes([360, 820])
        self.references.model().rowsInserted.connect(self.update_mode)
        self.references.model().rowsRemoved.connect(self.update_mode)
        self.update_mode()
        self.statusBar().showMessage(
            "Choose a local image folder. Add references for identity matching."
        )

    def update_mode(self, *_):
        matching = self.references.count() > 0
        self.model.setEnabled(matching)
        self.custom_threshold.setEnabled(matching)
        self.threshold.setEnabled(matching and self.custom_threshold.isChecked())
        self.copy_matches.setEnabled(matching)
        self.mode_label.setText(
            "Mode: reference-photo matching" if matching else "Mode: face detection only"
        )
        for column in (3, 4, 5):
            self.table.setColumnHidden(column, not matching)

    def choose_folder(self, field):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
        if folder:
            field.setText(folder)

    def add_references(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Reference photos", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
        )
        existing = {
            self.references.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.references.count())
        }
        for path in paths:
            if path not in existing:
                item = QListWidgetItem(QIcon(path), Path(path).name)
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setToolTip(path)
                self.references.addItem(item)
                existing.add(path)

    def restore_settings(self):
        for key, field in [("source", self.source), ("output", self.output)]:
            field.setText(str(self.settings.value(key, field.text())))
        self.recursive.setChecked(self.settings.value("recursive", True, type=bool))
        self.copy_matches.setChecked(self.settings.value("copy_matches", True, type=bool))
        self.model.setCurrentText(str(self.settings.value("model", "SFace")))
        self.detector.setCurrentText(str(self.settings.value("detector", "opencv")))

    def save_settings(self):
        for key, value in [
            ("source", self.source.text()),
            ("output", self.output.text()),
            ("recursive", self.recursive.isChecked()),
            ("copy_matches", self.copy_matches.isChecked()),
            ("model", self.model.currentText()),
            ("detector", self.detector.currentText()),
        ]:
            self.settings.setValue(key, value)

    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return
        try:
            if not self.source.text().strip() or not self.output.text().strip():
                raise ValueError("Choose an image folder and an output folder.")
            config = Config(
                source=Path(self.source.text()).expanduser().resolve(),
                output=Path(self.output.text()).expanduser().resolve(),
                references=[
                    Path(self.references.item(index).data(Qt.ItemDataRole.UserRole))
                    for index in range(self.references.count())
                ],
                recursive=self.recursive.isChecked(),
                model=self.model.currentText(),
                detector=self.detector.currentText(),
                threshold=self.threshold.value() if self.custom_threshold.isChecked() else None,
                copy_matches=self.copy_matches.isChecked(),
            )
            config.validate()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Check setup", str(exc))
            return
        self.save_settings()
        self.results.clear()
        self.table.setRowCount(0)
        self.log.clear()
        self.filter.setCurrentIndex(0)
        self.run_dir = None
        self.open_output.setEnabled(False)
        self.counts.setText("Found 0 · Processed 0 · Matches 0 · Errors 0")
        self.progress.setRange(0, 0)
        self.preview.pixmap = QPixmap()
        self.preview.image.clear()
        self.preview.image.setText("Finding image files…")
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
                "Stopping after the current model or file operation finishes…"
            )

    def handle_update(self, name, data):
        if name == "run_dir":
            self.run_dir = Path(data)
            self.open_output.setEnabled(True)
            self.log.appendPlainText(f"Saving reports to {data}")
        elif name == "status":
            self.preview.caption.setText(str(data))
            self.statusBar().showMessage(str(data))
        elif name == "preview":
            self.preview.show_image(data)
        elif name == "discovery":
            self.statusBar().showMessage(f"Finding image files… {data} found")
        elif name == "stats":
            self.counts.setText(
                f"Found {data['found']} · Processed {data['processed']} · "
                f"With faces {data['with_faces']} · Matches {data['matches']} · Errors {data['errors']}"
            )
            self.progress.setRange(0, max(1, data["found"]))
            self.progress.setValue(data["processed"])
            self.progress.setFormat(f"{data['processed']} / {data['found']}")
        elif name == "result":
            self.add_result(data)
        elif name in {"log", "error"}:
            self.log.appendPlainText(("ERROR: " if name == "error" else "") + str(data))
        elif name == "done":
            outcome = data["outcome"].replace("_", " ").title()
            self.progress.setRange(0, max(1, data["found"]))
            self.progress.setValue(data["processed"])
            self.progress.setFormat(outcome)
            self.preview.caption.setText(f"{outcome} · {data['processed']} images processed")
            self.statusBar().showMessage(
                f"{outcome}. {data['matches']} candidates; {data['with_faces']} images with faces."
            )
            self.log.appendPlainText(f"{outcome}. {data['remaining']} images left unprocessed.")

    def worker_finished(self):
        self.setup.setEnabled(True)
        self.update_mode()
        self.start.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.close_after_stop:
            QTimer.singleShot(0, self.close)

    def add_result(self, data):
        row = self.table.rowCount()
        self.results.append(data)
        self.table.insertRow(row)
        values = [
            data["relative_path"],
            data["status"].replace("_", " "),
            str(data.get("face_count", "—")),
            f"{data['distance']:.4f}" if data.get("distance") is not None else "—",
            f"{data['threshold']:.4f}" if data.get("threshold") is not None else "—",
            str(data.get("supporting_references", "—")),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0:
                item.setIcon(QIcon(data["source_path"]))
            item.setToolTip(data.get("error") or data["source_path"])
            if data["status"] == "candidate_match":
                item.setForeground(QColor("#5de4b4"))
            self.table.setItem(row, column, item)
        self.table.setRowHeight(row, 46)
        self.filter_results()

    def filter_results(self):
        selected = self.filter.currentData()
        for row, data in enumerate(self.results):
            show = (
                selected == "all"
                or data["status"] == selected
                or (selected == "with_faces" and data.get("face_count", 0) > 0)
                or (selected == "processing_error" and bool(data.get("error")))
            )
            self.table.setRowHidden(row, not show)

    def show_output(self):
        if self.run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.close_after_stop = True
            self.stop_scan()
            event.ignore()
        else:
            self.save_settings()
            event.accept()


STYLE = """
QMainWindow, QWidget { background: #151d2a; color: #e0e7f1; font-size: 12px; }
QGroupBox { border: 1px solid #324054; border-radius: 8px; margin-top: 13px; padding: 12px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #b5c6df; }
QLineEdit, QComboBox, QDoubleSpinBox, QListWidget, QPlainTextEdit, QTableWidget {
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
    app.setApplicationName("Folder Face Finder")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
