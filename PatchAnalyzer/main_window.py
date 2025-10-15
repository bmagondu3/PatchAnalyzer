# PatchAnalyzer/main_window.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5 import QtWidgets, QtCore # Import QtCore for QProgressDialog

from .models.data_loader import load_metadata
from .utils.log import setup_logger
from .views.VCanalysis_page import VCAnalysisPage
from .views.CCanalysis_page import CCAnalysisPage
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
        self.group.done.connect(self._open_voltage_analysis_page)

        if self.stack.count() < 2:
            self.stack.addWidget(self.group)
        else:
            if self.stack.count() > 1 and self.stack.widget(1) is not self.group:
                self.stack.removeWidget(self.stack.widget(1))
                self.stack.insertWidget(1, self.group)
            elif self.stack.count() < 2:
                self.stack.addWidget(self.group)

        self.stack.setCurrentWidget(self.group)

    # ------------------------------------------------------- Group → VC Analysis
    def _open_voltage_analysis_page(self, meta_df: pd.DataFrame):
        self.analysis = VCAnalysisPage(meta_df)

        self.analysis.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.group)
        )
        self.analysis.continue_requested.connect(
            lambda: self._open_cc_analysis_page(self.analysis.meta_df)
        )

        if self.stack.count() < 3:
            self.stack.addWidget(self.analysis)
        else:
            if self.stack.widget(2) is not None:
                self.stack.removeWidget(self.stack.widget(2))
            self.stack.insertWidget(2, self.analysis)

        self.stack.setCurrentWidget(self.analysis)

    # ------------------------------------------------------- VC Analysis → CC Analysis
    def _open_cc_analysis_page(self, meta_df: pd.DataFrame):
        # NEW: Loading screen for CC Analysis page
        progress_dialog = QtWidgets.QProgressDialog(
            "Loading Current Protocol data... Please wait.",
            None, 0, 0, self # Min/Max 0, so it's an indeterminate busy indicator
        )
        progress_dialog.setWindowTitle("Loading Data")
        progress_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        progress_dialog.setCancelButton(None) # No cancel button
        progress_dialog.setMinimumDuration(0) # Show immediately
        progress_dialog.show()
        QtWidgets.QApplication.processEvents() # Ensure dialog appears

        try:
            self.cc_analysis = CCAnalysisPage(meta_df)

            self.cc_analysis.back_requested.connect(
                lambda: self.stack.setCurrentWidget(self.analysis)
            )
            self.cc_analysis.continue_requested.connect(
                lambda: self._open_map_page(self.cc_analysis.meta_df)
            )

            if self.stack.count() < 4:
                self.stack.addWidget(self.cc_analysis)
            else:
                if self.stack.widget(3) is not None:
                    self.stack.removeWidget(self.stack.widget(3))
                self.stack.insertWidget(3, self.cc_analysis)

            self.stack.setCurrentWidget(self.cc_analysis)
        finally:
            progress_dialog.close() # Ensure dialog is closed even if error

    # ------------------------------------------------------- CC Analysis → Map
    def _open_map_page(self, meta_df: pd.DataFrame):
        self.map = MapPage(meta_df)
        self.map.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.cc_analysis)
        )

        if self.stack.count() < 5:
            self.stack.addWidget(self.map)
        else:
            if self.stack.widget(4) is not None:
                self.stack.removeWidget(self.stack.widget(4))
            self.stack.insertWidget(4, self.map)

        self.stack.setCurrentWidget(self.map)