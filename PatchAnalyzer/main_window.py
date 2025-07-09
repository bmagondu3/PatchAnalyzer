# PatchAnalyzer/main_window.py
from __future__ import annotations
from pathlib import Path
from PyQt5 import QtWidgets

from .utils.log import setup_logger
from .views.welcome_page import WelcomePage
from .views.main_page import MainPage
from .models.data_loader import load_metadata

logger = setup_logger(__name__)


class PatchAnalyzerGUI(QtWidgets.QMainWindow):
    """Top-level window that swaps WelcomePage ⇆ MainPage."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchAnalyzer")
        self.resize(1200, 600)

        self.stack = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.welcome = WelcomePage()
        self.welcome.select_folders.connect(self._on_folders_chosen)
        self.stack.addWidget(self.welcome)

    # ------------------------------------------------------- callbacks
    def _on_folders_chosen(self, dirs: list[Path]):
        try:
            meta_df = load_metadata(dirs)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "No valid data", str(exc))
            return

        self.main = MainPage(meta_df)
        if self.stack.count() == 1:
            self.stack.addWidget(self.main)
        else:                              # hot-reload if user goes back later
            self.stack.removeWidget(self.stack.widget(1))
            self.stack.addWidget(self.main)
        self.stack.setCurrentIndex(1)

