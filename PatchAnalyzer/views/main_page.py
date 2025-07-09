# PatchAnalyzer/views/main_page.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils.helpers import find_voltage_image
from ..utils.log import setup_logger

logger = setup_logger(__name__)


class MainPage(QtWidgets.QWidget):
    """Page 1 – scatter map (left) + cell/voltage images (right)."""

    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df.reset_index(drop=True)

        self.unique_coords = (
            self.meta_df[["stage_x", "stage_y", "stage_z"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.coord_to_repr = (
            self.meta_df.groupby(["stage_x", "stage_y", "stage_z"])
            .head(1)
            .reset_index()
            .set_index(["stage_x", "stage_y", "stage_z"])["index"]
            .to_dict()
        )

        self._build_ui()
        self._populate_scatter()

    # ------------------------------------------------------------------ UI
# ------------------------------------------------------------------ UI
    def _build_ui(self):
        base = QtWidgets.QHBoxLayout(self)
        base.setContentsMargins(8, 8, 8, 8)

        # ── LEFT : scatter plot ──────────────────────────────────────────
        self.plot = pg.PlotWidget(background="w")
        base.addWidget(self.plot, 1)            # stretch factor 1

        # ── RIGHT : images (top-half) + big buttons (bottom-half) ───────
        right = QtWidgets.QVBoxLayout()
        base.addLayout(right, 1)                # stretch factor 1 (same width)

        # ── TOP-half (images) ───────────────────────────────────────────
        imgs = QtWidgets.QHBoxLayout()
        right.addLayout(imgs, stretch=1)        # stretch 1  → top 50 %

        self.label_cell = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label_cell.setFrameShape(QtWidgets.QFrame.Box)
        self.label_cell.setMinimumSize(300, 300)
        imgs.addWidget(self.label_cell, 1)

        self.label_voltage = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label_voltage.setFrameShape(QtWidgets.QFrame.Box)
        self.label_voltage.setMinimumSize(300, 300)
        imgs.addWidget(self.label_voltage, 1)

        # ── BOTTOM-half (stacked buttons) ────────────────────────────────
        btn_box = QtWidgets.QVBoxLayout()
        right.addLayout(btn_box, stretch=1)     # stretch 1  → bottom 50 %

        btn_style = "font-size: 24px; padding: 20px;"           # ⤴ bigger
        self.btn_group = QtWidgets.QPushButton("Group Cells ▶")
        self.btn_group.setStyleSheet(btn_style)
        self.btn_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                    QtWidgets.QSizePolicy.Expanding)
        btn_box.addWidget(self.btn_group, 1)    # equal share of 50 %

        self.btn_skip = QtWidgets.QPushButton("Skip Grouping ➜")
        self.btn_skip.setStyleSheet(btn_style)
        self.btn_skip.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                    QtWidgets.QSizePolicy.Expanding)
        btn_box.addWidget(self.btn_skip, 1)     # stacked under “Group”


    # ---------------------------------------------------------- scatter map
    def _populate_scatter(self):
        """Create one dot for every row in meta_df."""
        spots = [
            dict(
                pos=(row.stage_x, row.stage_y),
                data=int(i),                     # store dataframe row index
                size=10,
                symbol="o",
                brush=pg.mkBrush(30, 144, 255, 150),
                pen=pg.mkPen(None),
            )
            for i, row in self.meta_df.iterrows()
        ]

        self.scatter = pg.ScatterPlotItem()
        self.scatter.addPoints(spots)
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.plot.addItem(self.scatter)
            # Optional: initial label text
        self.label_cell.setText("Select a cell")
        self.label_voltage.setText("Select a cell")

    # ---------------------------------------------------- event: point click
    def _on_point_clicked(self, _, points):
        logger.debug("🖱  clicked! %s", points[0].pos())
        df_idx = int(points[0].data())          # DataFrame row index
        row    = self.meta_df.loc[df_idx]

        self._show_cell_image(row)
        self._show_voltage_image(row)

        # highlight selection
        for p in self.scatter.points():
            p.resetPen()
        points[0].setPen(pg.mkPen(width=2, color="g"))
    # --------------------------------------------------- helper: images
    def _show_cell_image(self, row):
        self.label_cell.clear()  
        self.label_cell.setPixmap(QtGui.QPixmap())
        img_path = row["src_dir"] / "CellMetadata" / row["image"]
        logger.debug("CELL → %s", img_path)
        pm = QtGui.QPixmap(str(img_path))
        if not pm.isNull():
            self.label_cell.setPixmap(
                pm.scaled(self.label_cell.size(), QtCore.Qt.KeepAspectRatio)
            )
        else:
            self.label_cell.setText("No image")

    def _show_voltage_image(self, row):
        self.label_voltage.clear()  
        self.label_voltage.setPixmap(QtGui.QPixmap()) 
        png = find_voltage_image(row["src_dir"], row["image"])
        logger.debug("VOLT → %s", png) 
        if png:
            pm = QtGui.QPixmap(str(png))
            self.label_voltage.setPixmap(
                pm.scaled(self.label_voltage.size(), QtCore.Qt.KeepAspectRatio)
            )
        else:
            self.label_voltage.setText("No voltage image")

