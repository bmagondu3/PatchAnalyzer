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
    """Three always-square panes + fixed-height button bar (no grey gaps)."""

    group_requested = QtCore.pyqtSignal(pd.DataFrame)
    back_requested  = QtCore.pyqtSignal()

    # ------------------------------------------------------------------ init
    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df.reset_index(drop=True)
        self._frames: list[QtWidgets.QFrame] = []           # panes to resize
        self._btn_bar: QtWidgets.QWidget | None = None      # will hold buttons

        self._build_ui()
        self._populate_scatter()

    # -------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.setContentsMargins(0, 0, 0, 0)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # helper → bordered square frame for one pane
        def square_frame(widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
            frame = QtWidgets.QFrame()
            frame.setFrameShape(QtWidgets.QFrame.Box)
            frame.setLineWidth(1)
            frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                QtWidgets.QSizePolicy.Expanding)

            lay = QtWidgets.QVBoxLayout(frame)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(widget)
            self._frames.append(frame)
            return frame

        # ── TOP ROW : 3 panes ──────────────────────────────────────────
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(0)
        root.addLayout(top, stretch=1)

        # scatter
        self.plot = pg.PlotWidget(background="w")
        self.plot.getPlotItem().getViewBox().setAspectLocked(True)
        top.addWidget(square_frame(self.plot))

        # cell
        self.label_cell = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        top.addWidget(square_frame(self.label_cell))

        # voltage
        self.label_voltage = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        top.addWidget(square_frame(self.label_voltage))

        # top.addStretch(1)   # tiny residual width goes to the right edge

        # ── BOTTOM : fixed-height button bar ───────────────────────────
        self._btn_bar = QtWidgets.QWidget()
        bar_lay = QtWidgets.QHBoxLayout(self._btn_bar)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(0)

        bar_lay.addStretch(1)

        
        self.btn_back = QtWidgets.QPushButton("← Back")
        bar_lay.addWidget(self.btn_back)

        self.btn_group = QtWidgets.QPushButton("Group Cells ▶")
        bar_lay.addWidget(self.btn_group)

        # lock bar height to its sizeHint so it never stretches
        self._btn_bar.setFixedHeight(self._btn_bar.sizeHint().height())
        root.addWidget(self._btn_bar)

        # wire-up buttons
        self.btn_group.clicked.connect(self._on_group_clicked)
        self.btn_back.clicked.connect(self.back_requested.emit)

    # ---------------------------------------------------------------- scatter
    def _populate_scatter(self) -> None:
        spots = [
            dict(pos=(row.stage_x, row.stage_y), data=int(i),
                 size=10, symbol="o",
                 brush=pg.mkBrush(30, 144, 255, 150),
                 pen=pg.mkPen(None))
            for i, row in self.meta_df.iterrows()
        ]
        self.scatter = pg.ScatterPlotItem()
        self.scatter.addPoints(spots)
        self.scatter.sigClicked.connect(self._on_point_clicked)
        self.plot.addItem(self.scatter)

        self.label_cell.setText("Select a cell")
        self.label_voltage.setText("Select a cell")

    # ---------------------------------------------------------------- resize
    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        """Resize all three panes to the same square, consuming all height."""
        bar_h  = self._btn_bar.height()
        avail_h = self.height() - bar_h
        avail_w = self.width()
        side = int(min(avail_h, avail_w / 3))

        for f in self._frames:
            f.setFixedSize(side, side)

        super().resizeEvent(ev)

    # ---------------------------------------------------------------- click
    def _on_point_clicked(self, _, pts):
        idx = int(pts[0].data())
        row = self.meta_df.loc[idx]

        self._show(self.label_cell,
                   Path(row["src_dir"]) / "CellMetadata" / row["image"])
        self._show(self.label_voltage,
                   find_voltage_image(Path(row["src_dir"]), row["image"]))

        for p in self.scatter.points():
            p.resetPen()
        pts[0].setPen(pg.mkPen(width=2, color="g"))

    # ---------------------------------------------------------------- utils
    def _show(self, lbl: QtWidgets.QLabel, path: Path | None) -> None:
        lbl.clear()
        if path and (pm := QtGui.QPixmap(str(path))) and not pm.isNull():
            lbl.setPixmap(pm.scaled(
                lbl.size(), QtCore.Qt.KeepAspectRatioByExpanding,
                QtCore.Qt.SmoothTransformation))
        else:
            lbl.setText("No image")

    def _on_group_clicked(self):
        self.group_requested.emit(self.meta_df)
