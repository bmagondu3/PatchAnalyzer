# PatchAnalyzer/views/analysis_page.py
from __future__ import annotations
import pandas as pd
from PyQt5 import QtCore, QtWidgets


class AnalysisPage(QtWidgets.QWidget):
    """Placeholder – will host the actual analysis later."""
    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(
            QtWidgets.QLabel(
                "<h2>Analysis page coming soon…</h2>"
                f"<p>Rows in DataFrame: {len(self.meta_df)}<br>"
                f"Grouped rows: {self.meta_df['group_label'].ne('').sum()}</p>",
                alignment=QtCore.Qt.AlignCenter
            ),
            1
        )
