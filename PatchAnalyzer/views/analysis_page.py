# PatchAnalyzer/views/analysis_page.py
from __future__ import annotations
from pathlib import Path
import re, hashlib

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..models.ephys_loader import load_voltage_traces_for_indices
from ..utils.passives import compute_passive_params


# ── helper – extract numeric cell‑ID from image filename (e.g. cell_7.webp) ──
_ID_RE = re.compile(r"cell[_\-]?(\d+)", re.IGNORECASE)

def _cell_id(img_name: str) -> int | None:
    m = _ID_RE.search(img_name)
    return int(m.group(1)) if m else None


class AnalysisPage(QtWidgets.QWidget):
    """Left‑hand cell table  •  Command/Response stacked plots  •  Bottom controls."""

    # ------------------------------ signals (mirrors GroupPage naming) ----
    back_requested      = QtCore.pyqtSignal()
    analyze_requested   = QtCore.pyqtSignal(pd.DataFrame)   # emits meta_df when ▶ Analyze
    continue_requested  = QtCore.pyqtSignal()

    # ---------------------------------------------------------------- init
    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df.reset_index(drop=True)

        # ── aggregate identical coordinates → list[cell_ids] ──────────────
        self._cells: list[dict] = []
        for coord, sub in self.meta_df.groupby(["stage_x", "stage_y", "stage_z"]):
            cell_ids = [_cell_id(img) for img in sub["image"] if _cell_id(img) is not None]
            self._cells.append(
                dict(coord=coord,
                     src_dir=Path(sub["src_dir"].iloc[0]),
                     group_label=sub["group_label"].iloc[0],
                     cell_ids=sorted(cell_ids)))

        # ── widgets ------------------------------------------------------
        self.table = QtWidgets.QTableWidget(
            selectionBehavior=QtWidgets.QAbstractItemView.SelectRows,
            selectionMode=QtWidgets.QAbstractItemView.SingleSelection,
        )
        self.table.verticalHeader().hide()

        # right‑hand plots (command over response) -----------------------
        right_box = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self.plot_cmd = pg.PlotWidget(background="w")
        self.plot_cmd.setMaximumHeight(180)
        self.plot_cmd.getPlotItem().hideAxis('bottom')      # share x‑axis with lower plot
        right_lay.addWidget(self.plot_cmd, 1)

        self.plot_rsp = pg.PlotWidget(background="w")
        self.legend = self.plot_rsp.addLegend(offset=(-110, 30))
        right_lay.addWidget(self.plot_rsp, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(1, 1)

        # bottom button bar --------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        self.btn_save_csv = QtWidgets.QPushButton("Save Full CSV…")
        bar.addWidget(self.btn_save_csv)
        self.btn_save_csv.clicked.connect(self._save_full_csv)   # NEW


        self.btn_analyze = QtWidgets.QPushButton("Analyze ▶")
        bar.addWidget(self.btn_analyze)
        self.btn_analyze.clicked.connect(self._on_analyze)
        bar.addStretch(1)

        self.chk_show_cmd = QtWidgets.QCheckBox("Show Command")
        self.chk_show_cmd.setChecked(True)
        bar.addWidget(self.chk_show_cmd)

        self.btn_back = QtWidgets.QPushButton("← Back")
        bar.addWidget(self.btn_back)
        self.btn_back.clicked.connect(self.back_requested)



        self.btn_continue = QtWidgets.QPushButton("Continue ▶")
        bar.addWidget(self.btn_continue)
        self.btn_continue.clicked.connect(self.continue_requested)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(bar)

        self.chk_show_cmd.toggled.connect(self.plot_cmd.setVisible)

        self._curve_pairs: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []  # (cmd, rsp)
        self._selected_pair: tuple[pg.PlotDataItem, pg.PlotDataItem] | None = None

        self._populate_table()
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)

    # ---------------------------------------------------- populate table --
    _COLS = ["indices", "src_dir", "group_label",
            "Ra (MΩ)", "Rm (MΩ)", "Cm (pF)"]


    def _populate_table(self):
        self.table.setColumnCount(len(self._COLS))
        self.table.setHorizontalHeaderLabels([c.replace("_", " ").title() for c in self._COLS])
        self.table.setRowCount(len(self._cells))

        for r, cell in enumerate(self._cells):
            vals = {
                "indices": ", ".join(map(str, cell["cell_ids"])),
                "src_dir": cell["src_dir"].name,
                "group_label": cell["group_label"],
                "Ra (MΩ)": "",
                "Rm (MΩ)": "",
                "Cm (pF)": "",
            }
            for c, col in enumerate(self._COLS):
                item = QtWidgets.QTableWidgetItem(vals[col])
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                if col == "group_label" and vals[col]:
                    hue = int(hashlib.md5(vals[col].encode()).hexdigest(), 16) % 360
                    item.setBackground(QtGui.QColor.fromHsl(hue, 160, 200))
                    item.setForeground(QtGui.QColor("black"))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, c, item)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)
            self._on_row_selected()

    # -------------------------------------------------- row selection ----
    def _on_row_selected(self, *_):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        cell = self._cells[sel[0].row()]

        # clear previous curves / legend
        for p in (self.plot_cmd, self.plot_rsp):
            p.clear()
        self.legend.clear()
        self._curve_pairs.clear()
        self._selected_pair = None

        traces = load_voltage_traces_for_indices(cell["src_dir"], cell["cell_ids"])
        if not traces:
            self.plot_rsp.addItem(pg.TextItem(html="<span style='color:red'>No traces found</span>"))
            return

        for fname, (t, cmd, rsp) in traces.items():
            pen = pg.mkPen("red", width=1)
            curve_cmd = self.plot_cmd.plot(t, cmd, pen=pen, clickable=True)
            curve_rsp = self.plot_rsp.plot(t, rsp, pen=pen, clickable=True, name=fname)
            curve_rsp.sigClicked.connect(lambda _, c1=curve_cmd, c2=curve_rsp: self._on_pair_clicked(c1, c2))
            curve_cmd.sigClicked.connect(lambda _, c1=curve_cmd, c2=curve_rsp: self._on_pair_clicked(c1, c2))

            # legend toggle visibility -----------------------------------
            label_item = self.legend.items[-1][1]
            label_item.setAcceptHoverEvents(True)
            def _make_toggle(c1=curve_cmd, c2=curve_rsp, lbl=label_item):
                def _toggle(_):
                    vis = not c1.isVisible()
                    for cv in (c1, c2):
                        cv.setVisible(vis)
                    lbl.setOpacity(1.0 if vis else 0.3)
                return _toggle
            label_item.mousePressEvent = _make_toggle()

            self._curve_pairs.append((curve_cmd, curve_rsp))

        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())
        self.plot_cmd.enableAutoRange()
        self.plot_rsp.enableAutoRange()

    # ------------------------------------------------- curve pair click ---
    def _on_pair_clicked(self, cmd_curve: pg.PlotDataItem, rsp_curve: pg.PlotDataItem):
        pair = (cmd_curve, rsp_curve)
        if self._selected_pair == pair:
            self._selected_pair = None
            for c1, c2 in self._curve_pairs:
                if c1.isVisible():
                    for cv in (c1, c2):
                        cv.setOpacity(1.0)
                        cv.setPen("red", width=1)
        else:
            self._selected_pair = pair
            for c1, c2 in self._curve_pairs:
                if (c1, c2) == pair and c1.isVisible():
                    for cv in (c1, c2):
                        cv

    # ─────────────────────────────────────────────────────────── analysis slot
    def _on_analyze(self):
        """
        Loop over every cell, run passive‑parameter extraction for every trace,
        take the mean, and fill the table.  Recomputes each time.
        """
        total = len(self._cells)
        dlg = QtWidgets.QProgressDialog("Analyzing passive parameters…",
                                        None, 0, total, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        for row_idx, cell in enumerate(self._cells):
            dlg.setLabelText(f"Cell {row_idx + 1} / {total}")
            QtWidgets.QApplication.processEvents()

            traces = load_voltage_traces_for_indices(cell["src_dir"],
                                                    cell["cell_ids"])
            ra, rm, cm = self._analyze_cell(traces)

            # cache for possible later use
            cell["Ra"] = ra
            cell["Rm"] = rm
            cell["Cm"] = cm

            self._update_table_row(row_idx, ra, rm, cm)

            dlg.setValue(row_idx + 1)
            if dlg.wasCanceled():
                break
        dlg.close()

        # keep the original outward signal so nothing else breaks
        self.analyze_requested.emit(self.meta_df)

    # -------------------------------------------------------------------------
    def _analyze_cell(self, traces: dict):
        """Return mean Ra, Rm, Cm for one cell (None if all fits fail)."""
        ra_list, rm_list, cm_list = [], [], []
        for t, cmd, rsp in traces.values():
            out = compute_passive_params(t, cmd, rsp, step_mV=10.0)
            if all(val is not None for val in out):
                ra_list.append(out[0])
                rm_list.append(out[1])
                cm_list.append(out[2])

        if not ra_list:          # no successful fits
            return None, None, None

        import numpy as np
        return float(np.mean(ra_list)), float(np.mean(rm_list)), float(np.mean(cm_list))

    # -------------------------------------------------------------------------
    def _update_table_row(self, row: int, ra, rm, cm):
        """Put formatted values into the passive‑parameter columns."""
        def _fmt(v):   # show en dash if None
            return f"{v:.1f}" if v is not None else "–"

        base_col = self._COLS.index("Ra (MΩ)")
        for offset, val in enumerate((_fmt(ra), _fmt(rm), _fmt(cm))):
            item = QtWidgets.QTableWidgetItem(val)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, base_col + offset, item)

    # ─────────────────────────────────────────────── CSV export
    def _save_full_csv(self):
        """
        Build a DataFrame that contains one row per *cell* with all metadata
        + passive parameters, then let the user save it as a CSV.
        """
        # Assemble a list of dictionaries – one per cell
        rows = []
        for cell in self._cells:
            rows.append({
                "indices"   : ", ".join(map(str, cell["cell_ids"])),
                "stage_x"   : cell["coord"][0],
                "stage_y"   : cell["coord"][1],
                "stage_z"   : cell["coord"][2],
                "src_dir"   : cell["src_dir"].name,
                "group_label": cell["group_label"],
                "Ra_MOhm"   : cell.get("Ra"),
                "Rm_MOhm"   : cell.get("Rm"),
                "Cm_pF"     : cell.get("Cm"),
            })

        df_export = pd.DataFrame(rows)

        # Ask for destination
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            df_export.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Analysis table successfully saved to:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Could not save CSV:\n{exc}"
            )
