# patch_gui/views/welcome_page.py
from __future__ import annotations
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

class WelcomePage(QtWidgets.QWidget):
    """Page 0 – welcome banner and *Select Data Folders* button."""
    select_folders = QtCore.pyqtSignal(list)      # emits list[Path]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Welcome to PatchAnalyzer!", self)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:32px;")
        lay.addWidget(title, 3)

        btn = QtWidgets.QPushButton("Select Data Folders", self)
        btn.setIcon(QtGui.QIcon.fromTheme("folder-open"))
        btn.clicked.connect(self._pick_folders)
        lay.addWidget(btn, 1, QtCore.Qt.AlignCenter)

        lay.setContentsMargins(80, 60, 80, 60)

    # ----------------------------------------------------------- folder pick
    def _pick_folders(self):
        """
        QFileDialog hack: enable **multi-directory** selection by disabling the
        native dialog and promoting both QListView & QTreeView to Multi-select
        mode. :contentReference[oaicite:0]{index=0}
        """
        dlg = QtWidgets.QFileDialog(self, "Select one or more experiment folders")
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)

        for cls in (QtWidgets.QListView, QtWidgets.QTreeView):
            for view in dlg.findChildren(cls):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.select_folders.emit([Path(p) for p in dlg.selectedFiles()])
