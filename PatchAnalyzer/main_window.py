# PatchAnalyzer/main_window.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets

from .models.data_loader import load_metadata
from .utils.log import setup_logger
from .views.analysis_page import AnalysisPage
from .views.group_page import GroupPage
from .views.main_page import MainPage
from .views.welcome_page import WelcomePage

logger = setup_logger(__name__)


class PatchAnalyzerGUI(QtWidgets.QMainWindow):
    """Top-level window: Welcome ↔ Main → Group → Analysis."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchAnalyzer")
        self.resize(1000, 450)

        self.stack = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stack)

        self.welcome = WelcomePage()
        self.welcome.select_folders.connect(self._on_folders_chosen)
        self.stack.addWidget(self.welcome)

    # ------------------------------------------------------- Welcome → Main
    def _on_folders_chosen(self, dirs: list[Path]):
        try:
            meta_df = load_metadata(dirs)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "No valid data", str(exc))
            return

        self.main = MainPage(meta_df)
        self.main.group_requested.connect(self._open_group_page)
        self.main.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.welcome)
        )

        if self.stack.count() == 1:
            self.stack.addWidget(self.main)
        else:
            self.stack.removeWidget(self.stack.widget(1))
            self.stack.addWidget(self.main)

        self.stack.setCurrentWidget(self.main)

    # ------------------------------------------------------- Main → Group
    def _open_group_page(self, meta_df: pd.DataFrame):
        self.group = GroupPage(meta_df)
        self.group.back_requested.connect(lambda: self.stack.setCurrentWidget(self.main))
        self.group.done.connect(self._open_analysis_page)

        if self.stack.count() < 3:
            self.stack.addWidget(self.group)
        else:
            self.stack.removeWidget(self.stack.widget(2))
            self.stack.addWidget(self.group)

        self.stack.setCurrentWidget(self.group)

    # ------------------------------------------------------- Group → Analysis
    def _open_analysis_page(self, meta_df: pd.DataFrame):
        self.analysis = AnalysisPage(meta_df)

        if self.stack.count() < 4:
            self.stack.addWidget(self.analysis)
        else:
            self.stack.removeWidget(self.stack.widget(3))
            self.stack.addWidget(self.analysis)

        self.stack.setCurrentWidget(self.analysis)
