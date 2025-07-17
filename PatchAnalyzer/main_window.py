# PatchAnalyzer/main_window.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets

from .models.data_loader import load_metadata
from .utils.log import setup_logger
from .views.VCanalysis_page import VCAnalysisPage
from .views.group_page import GroupPage
from .views.welcome_page import WelcomePage
from .views.map_page import MapPage

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

        # self.main = MainPage(meta_df)
        # self.main.group_requested.connect(self._open_group_page)
        # self.main.back_requested.connect(
        #     lambda: self.stack.setCurrentWidget(self.welcome)
        # )

        self._open_group_page(meta_df)


    # ------------------------------------------------------- Main → Group
    def _open_group_page(self, meta_df: pd.DataFrame):
        self.group = GroupPage(meta_df)
        self.group.back_requested.connect(lambda: self.stack.setCurrentWidget(self.welcome))
        self.group.done.connect(self._open_analysis_page)

        if self.stack.count() < 3:
            self.stack.addWidget(self.group)
        else:
            self.stack.removeWidget(self.stack.widget(2))
            self.stack.addWidget(self.group)

        self.stack.setCurrentWidget(self.group)

    # ------------------------------------------------------- Group → Analysis
    def _open_analysis_page(self, meta_df: pd.DataFrame):
        self.analysis = VCAnalysisPage(meta_df)

        # NEW ───────── connect “← Back” from VCnalysisPage to show GroupPage
        self.analysis.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.group)
        )
        self.analysis.continue_requested.connect(
            lambda: self._open_map_page(self.analysis.meta_df)
        )
        if self.stack.count() < 4:
            self.stack.addWidget(self.analysis)
        else:
            self.stack.removeWidget(self.stack.widget(3))
            self.stack.addWidget(self.analysis)

        self.stack.setCurrentWidget(self.analysis)

    # ------------------------------------------------------- Analysis → Map
    def _open_map_page(self, meta_df: pd.DataFrame):
        self.map = MapPage(meta_df)
        # Map’s ← Back returns to Analysis
        self.map.back_requested.connect(lambda: self.stack.setCurrentWidget(self.analysis))

        if self.stack.count() < 5:
            self.stack.addWidget(self.map)
        else:
            self.stack.removeWidget(self.stack.widget(4))
            self.stack.addWidget(self.map)

        self.stack.setCurrentWidget(self.map)
