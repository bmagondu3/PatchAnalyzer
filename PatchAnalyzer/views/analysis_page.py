# PatchAnalyzer/views/analysis_page.py
from __future__ import annotations
from pathlib import Path
import re, hashlib

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..models.ephys_loader import load_voltage_traces_for_indices
from ..utils.ephys_analyzer import VprotAnalyzer      # NEW import



# ── NEW single source-of-truth for columns ─────────────
_COL_IDS = [
    "unique_id", "indices", "src_dir", "group_label",        # ← src_dir added
    "mean_ra", "sd_ra", "mean_rm", "sd_rm", "mean_cm", "sd_cm",
    "ra_list", "rm_list", "cm_list",
    "has_vprot",
]

_COL_HEADERS = [
    "UID", "Indices", "Source Dir", "Group Label",           # ← header added
    "Mean Ra (MΩ)", "SD Ra (MΩ)",
    "Mean Rm (MΩ)", "SD Rm (MΩ)",
    "Mean Cm (pF)", "SD Cm (pF)",
    "Ra list (MΩ)", "Rm list (MΩ)", "Cm list (pF)",
    "Has Vprot",
]

# ───────────────────────────────────────────────────────

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
        self._vprot = VprotAnalyzer(step_mV=10.0)
        # ── aggregate identical coordinates → list[cell_ids] ──────────────
        # enumerate → give every cell a persistent UID
        self._cells: list[dict] = []
        for uid, (coord, sub) in enumerate(
                self.meta_df.groupby(["stage_x", "stage_y", "stage_z"]), start=1):
            cell_ids = [_cell_id(img) for img in sub["image"] if _cell_id(img) is not None]
            self._cells.append(dict(
                unique_id = uid,
                coord     = coord,
                src_dir   = Path(sub["src_dir"].iloc[0]),
                group_label = sub["group_label"].iloc[0],
                cell_ids  = sorted(cell_ids),
    ))
        # ── add near other instance attrs in __init__
        self._param_label: pg.TextItem | None = None
        self._param_conn  = None               # <— NEW

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
        right_lay.setSpacing(8)

        self.plot_cmd = pg.PlotWidget(background="w")
        self.plot_cmd.setMaximumHeight(180)
        self.plot_cmd.setLabel('left',   'Command (mV)')
        self.plot_cmd.setLabel('bottom', 'Time (ms)')
        right_lay.addWidget(self.plot_cmd, 1)

        self.plot_rsp = pg.PlotWidget(background="w")
        self.plot_rsp.setLabel('left',   'Response (pA)')
        self.plot_rsp.setLabel('bottom', 'Time (ms)')
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

    def _populate_table(self):
        self.table.setColumnCount(len(_COL_IDS))
        self.table.setHorizontalHeaderLabels(_COL_HEADERS)
        self.table.setRowCount(len(self._cells))
        # Map id → column index for quick use elsewhere
        self._col_idx = {cid: i for i, cid in enumerate(_COL_IDS)}

        for r, cell in enumerate(self._cells):
            init_vals = {
                "unique_id"  : str(cell["unique_id"]),
                "indices"    : ", ".join(map(str, cell["cell_ids"])),
                "src_dir"    : cell["src_dir"].name,          # NEW
                "group_label": cell["group_label"],
                # empty placeholders – filled after analysis
                "mean_ra": "", "sd_ra": "", "mean_rm": "", "sd_rm": "",
                "mean_cm": "", "sd_cm": "",
                "ra_list": "", "rm_list": "", "cm_list": "",
                "has_vprot": "",
            }

            for cid, text in init_vals.items():
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                if cid == "group_label" and text:
                    hue = int(hashlib.md5(text.encode()).hexdigest(), 16) % 360
                    item.setBackground(QtGui.QColor.fromHsl(hue, 160, 200))
                    item.setForeground(QtGui.QColor("black"))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, self._col_idx[cid], item)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        if self.table.rowCount():
            self.table.selectRow(0)
            self._on_row_selected()

    # -------------------------------------------------- row selection ----
    # ─────────────────────────────────────────────────────────── parameter label
    def _show_param_label(self, cell: dict):
        """Draw ‘Ra Rm Cm’ legend once per row-selection – no live callbacks."""
        # remove previous label
        if getattr(self, "_param_label", None):
            self.plot_rsp.removeItem(self._param_label)
        self._param_label = None

        # analysis not run → nothing to show
        if not all(k in cell for k in ("mean_ra", "mean_rm", "mean_cm")):
            return

        txt = (f"Ra: {cell['mean_ra']} MΩ    "
            f"Rm: {cell['mean_rm']} MΩ    "
            f"Cm: {cell['mean_cm']} pF")

        lbl = pg.TextItem(txt, anchor=(0, 1), color=(0, 0, 0))
        self.plot_rsp.addItem(lbl)
        self._param_label = lbl

        # position once (3 % inset from bottom-left corner)
        (xmin, xmax), (ymin, ymax) = self.plot_rsp.viewRange()
        lbl.setPos(xmin + 0.03 * (xmax - xmin),
                ymin + 0.03 * (ymax - ymin))

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

        # NEW: mapping response→command & ensure selected‑curve attr exists
        self._resp2cmd: dict[pg.PlotDataItem, pg.PlotDataItem] = {}
        self._selected_curve: pg.PlotDataItem | None = None

        traces = load_voltage_traces_for_indices(cell["src_dir"], cell["cell_ids"])
        if not traces:
            self.plot_rsp.addItem(pg.TextItem(html="<span style='color:red'>No traces found</span>"))
            return

        from functools import partial   # avoid late‑binding lambda issue

        for fname, (t, cmd, rsp) in traces.items():
            t_ms   = t * 1e3        # s → ms
            cmd_mV = cmd * 1e3      # V → mV
            rsp_pA = rsp * 1e12     # A → pA

            pen = pg.mkPen("red", width=1)
            curve_cmd = self.plot_cmd.plot(t_ms, cmd_mV, pen=pen, clickable=True)
            curve_rsp = self.plot_rsp.plot(t_ms, rsp_pA, pen=pen, clickable=True,name=fname)


            # connect clicks – pass ONLY the response curve; event arg ignored
            curve_rsp.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))
            curve_cmd.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))

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
            self._resp2cmd[curve_rsp] = curve_cmd        # populate mapping

        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())
        self.plot_cmd.enableAutoRange()
        self.plot_rsp.enableAutoRange()
            # -------- show Ra / Rm / Cm text -----------
        self._show_param_label(cell)


    # ------------------------------------------------ response click ------
    def _on_response_clicked(self, rsp_curve: pg.PlotDataItem, _event=None):
        """
        Highlight the sweep whose *response* curve was clicked.
        """
        cmd_curve = self._resp2cmd[rsp_curve]

        # toggle selection
        self._selected_curve = rsp_curve if self._selected_curve is not rsp_curve else None

        for r_curve, c_curve in self._resp2cmd.items():
            visible = r_curve.isVisible() or c_curve.isVisible()
            selected = (r_curve is self._selected_curve) and visible

            if selected:
                opacity, width = 1.0, 3
            elif visible:
                opacity, width = 0.25, 1
            else:                     # hidden via legend
                opacity, width = 0.05, 1

            for cv in (r_curve, c_curve):
                cv.setOpacity(opacity)
                cv.setPen("red", width=width)



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

            traces = load_voltage_traces_for_indices(
                cell["src_dir"], cell["cell_ids"])

            stats = self._analyze_cell(traces)

            # keep for CSV export
            cell.update(stats)

            self._update_table_row(row_idx, stats)

            dlg.setValue(row_idx + 1)
            if dlg.wasCanceled():
                break
        dlg.close()

                # refresh the currently-selected row’s plots / stats  ← NEW
        if self.table.currentRow() >= 0:
            self._on_row_selected()


        # keep the original outward signal so nothing else breaks
        self.analyze_requested.emit(self.meta_df)

    # -------------------------------------------------------------------------
    def _analyze_cell(self, traces: dict):
        """
        Return a dict with all stats for one cell.
        Keys match _COL_IDS; values are str-ready.
        """
        stats = {cid: "–" for cid in (
            "mean_ra", "sd_ra", "mean_rm", "sd_rm", "mean_cm", "sd_cm",
            "ra_list", "rm_list", "cm_list")}
        stats["has_vprot"] = "✓" if traces else "–"

        if not traces:
            return stats          # nothing to fit

        (means, fits) = self._vprot.fit_cell(
            traces, aggregate="mean", return_all=True)

        ok = [f for f in fits if all(v is not None for v in f)]
        if not ok:
            return stats          # all fits failed

        import numpy as np
        ra, rm, cm = zip(*ok)
        def _fmt(v): return f"{v:.1f}"
        def _sd(lst): return np.std(lst, ddof=1) if len(lst) > 1 else 0.0

        stats.update(
            mean_ra = _fmt(np.mean(ra)),  sd_ra = _fmt(_sd(ra)),
            mean_rm = _fmt(np.mean(rm)),  sd_rm = _fmt(_sd(rm)),
            mean_cm = _fmt(np.mean(cm)),  sd_cm = _fmt(_sd(cm)),
            ra_list = ", ".join(_fmt(v) for v in ra),
            rm_list = ", ".join(_fmt(v) for v in rm),
            cm_list = ", ".join(_fmt(v) for v in cm),
        )
        return stats


    # -------------------------------------------------------------------------
    def _update_table_row(self, row: int, stats: dict):
        for cid, val in stats.items():
            if cid not in self._col_idx:
                continue
            item = QtWidgets.QTableWidgetItem(val)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self._col_idx[cid], item)


    # ─────────────────────────────────────────────── CSV export
    def _save_full_csv(self):
        rows = []
        for cell in self._cells:
            rows.append({
                "UID"        : cell["unique_id"],
                "indices"    : ", ".join(map(str, cell["cell_ids"])),
                "stage_x"    : cell["coord"][0],
                "stage_y"    : cell["coord"][1],
                "stage_z"    : cell["coord"][2],
                "src_dir"    : cell["src_dir"].name,
                "group_label": cell["group_label"],
                "mean_Ra_MOhm": cell.get("mean_ra", "–"),
                "SD_Ra_MOhm"  : cell.get("sd_ra", "–"),
                "mean_Rm_MOhm": cell.get("mean_rm", "–"),
                "SD_Rm_MOhm"  : cell.get("sd_rm", "–"),
                "mean_Cm_pF"  : cell.get("mean_cm", "–"),
                "SD_Cm_pF"    : cell.get("sd_cm", "–"),
                "Ra_list_MOhm": cell.get("ra_list", ""),
                "Rm_list_MOhm": cell.get("rm_list", ""),
                "Cm_list_pF"  : cell.get("cm_list", ""),
                "has_Vprot"   : cell.get("has_vprot", "–"),
            })
        pd.DataFrame(rows).to_csv(
            QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Analysis as CSV", "",
                "CSV Files (*.csv);;All Files (*)")[0],
            index=False)

