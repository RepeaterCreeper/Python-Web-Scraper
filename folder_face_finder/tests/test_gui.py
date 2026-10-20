import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLineEdit

from face_finder import gui


def test_visibility_controls_apply_to_existing_and_new_results(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda *_: settings)
    window = gui.MainWindow()
    data = {
        "relative_path": "private/person.png",
        "source_path": str(tmp_path / "person.png"),
        "status": "no_face",
        "face_count": 0,
    }
    window.add_result(data)
    window.hide_urls.setChecked(True)
    window.hide_thumbnails.setChecked(True)
    window.add_result(data)
    assert window.preview.isHidden()
    assert window.log.isHidden()
    assert window.source.echoMode() == QLineEdit.EchoMode.Password
    for row in range(2):
        assert window.table.item(row, 0).text() == f"Image {row + 1}"
        assert window.table.item(row, 0).icon().isNull()
        assert window.table.item(row, 0).toolTip() == ""
    window.workers.setValue(3)
    window.save_settings()
    restored = gui.MainWindow()
    assert restored.workers.value() == 3
    assert restored.hide_urls.isChecked()
    assert restored.hide_thumbnails.isChecked()
    window.hide_urls.setChecked(False)
    window.hide_thumbnails.setChecked(False)
    assert not window.preview.isHidden()
    assert window.table.item(0, 0).text() == data["relative_path"]
    assert window.table.item(0, 0).toolTip() == data["source_path"]
    window.close()
    restored.close()
    app.processEvents()
