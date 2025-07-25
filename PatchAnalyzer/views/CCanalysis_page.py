# PatchAnalyzer/views/CCAnalysis_page.py
from __future__ import annotations
from pathlib import Path
import re, hashlib

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from ..utils.ephys_analyzer import CprotAnalyzer          # ← NEW
import numpy as np                                        # ← NE
from ..models.data_loader import load_current_sweeps, load_current_traces

_COL_IDS = [
    "unique_id", "indices", "src_dir", "group_label",
    "mean_rmp", "mean_tau", "mean_rm", "mean_cm",
    "rmp_list", "tau_list", "rm_list", "cm_list",
    "has_cc",
]

_COL_HEADERS = [
    "UID", "Indices", "Source Dir", "Group Label",
    "Mean RMP (mV)", "Mean Tau (ms)",
    "Mean Rm (MΩ)", "Mean Cm (pF)",
    "RMP (list)", "Tau (list)", "Rm (list)", "Cm (list)",
    "Has CC?",
]

_SWEEP_COLS = [
    "Cell(UniqueID)", "Index", "coordinates", "Source Dir", "Group Label",
    "Injected current (pA)", "RMP mV", "Tau ms", "Rm (Mohms)", "Cm (pF)",
    "Firing rate (Hz)", "mean AP peak mV", "mean AP hwdt(ms)",
    "threshold", "dV/dt max (mV/s)",
]

_ID_RE = re.compile(r"cell[_\-]?(\d+)", re.IGNORECASE)
_CP_CELL_ID_RE = re.compile(r"CurrentProtocol_(\d+)_#")

class CCAnalysisPage(QtWidgets.QWidget):
    """Left‑hand cell table  •  Command/Response stacked plots  •  Bottom controls."""

    back_requested      = QtCore.pyqtSignal()
    analyze_requested   = QtCore.pyqtSignal(pd.DataFrame)
    continue_requested  = QtCore.pyqtSignal()

    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df.reset_index(drop=True)
        self._cprot = CprotAnalyzer()  # ← NEW: Current Protocol Analyzer
        self._sweep_df: pd.DataFrame = pd.DataFrame(columns=_SWEEP_COLS)

        # build cells exactly as in VCAnalysisPage – same UIDs & same image‑derived indices
        from .VCanalysis_page import _cell_id  # reuse the same helper

        self._cells: list[dict] = []
        for uid, (coord, sub) in enumerate(
                self.meta_df.groupby(["stage_x", "stage_y", "stage_z"]), start=1):
            src_dir_path = Path(sub["src_dir"].iloc[0])

            # derive cell‑IDs from the image filenames – exactly like VC
            image_ids = [_cell_id(img) for img in sub["image"]]
            image_ids = [i for i in image_ids if i is not None]
            image_ids = sorted(set(image_ids))

            self._cells.append(dict(
                unique_id   = uid,
                coord       = coord,
                src_dir     = src_dir_path,
                group_label = sub["group_label"].iloc[0],
                cell_ids    = image_ids,
            ))

        self._param_label: pg.TextItem | None = None
        self._param_conn    = None
        self._current_cell_index_display = 0

        self.table = QtWidgets.QTableWidget(
            selectionBehavior=QtWidgets.QAbstractItemView.SelectRows,
            selectionMode=QtWidgets.QAbstractItemView.SingleSelection,
        )
        self.table.verticalHeader().hide()

        right_box = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self.plot_cmd = pg.PlotWidget(background="w")
        self.plot_cmd.setMaximumHeight(180)
        self.plot_cmd.setLabel('left',   'Command (pA)')
        self.plot_cmd.setLabel('bottom', 'Time (ms)')
        right_lay.addWidget(self.plot_cmd, 1)

        self.plot_rsp = pg.PlotWidget(background="w")
        self.plot_rsp.setLabel('left',   'Response (mV)')
        self.plot_rsp.setLabel('bottom', 'Time (ms)')
        # place legend as a grid in the top-right
        self.legend = self.plot_rsp.addLegend(offset=(-10, 10))
        self.legend.setColumnCount(3)
        right_lay.addWidget(self.plot_rsp, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(1, 1)

        # bottom button bar --------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        self.btn_save_csv = QtWidgets.QPushButton("Save Full CSV…")
        bar.addWidget(self.btn_save_csv)
        self.btn_save_csv.clicked.connect(self._save_full_csv)

        self.btn_analyze = QtWidgets.QPushButton("Analyze ▶")
        bar.addWidget(self.btn_analyze)
        self.btn_analyze.clicked.connect(self._on_analyze)
        bar.addStretch(1)

        self.btn_previous_protocol = QtWidgets.QPushButton("Previous Protocol")
        self.btn_previous_protocol.clicked.connect(self._on_previous_protocol_clicked)
        bar.addWidget(self.btn_previous_protocol)

        self.btn_next_protocol = QtWidgets.QPushButton("Next Protocol")
        self.btn_next_protocol.clicked.connect(self._on_next_protocol_clicked)
        bar.addWidget(self.btn_next_protocol)

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
        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())

        self._curve_pairs = []
        self._selected_pair = None
        self._selected_curve = None
        self._resp2cmd: dict[pg.PlotDataItem, pg.PlotDataItem] = {}

        self._populate_table()
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)


    def _populate_table(self):
        self.table.setColumnCount(len(_COL_IDS))
        self.table.setHorizontalHeaderLabels(_COL_HEADERS)
        self.table.setRowCount(len(self._cells))
        self._col_idx = {cid: i for i, cid in enumerate(_COL_IDS)}

        for r, cell in enumerate(self._cells):
            indices_str = ", ".join(map(str, cell["cell_ids"]))
            init_vals = {
                "unique_id": str(cell["unique_id"]),
                "indices": indices_str,
                "src_dir": cell["src_dir"].name,
                "group_label": cell["group_label"],
                **{k: "" for k in ("mean_rmp","mean_tau","mean_rm","mean_cm","rmp_list","tau_list","rm_list","cm_list")},
                "has_cc": "",
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

    def _show_param_label(self, cell: dict):
        if getattr(self, "_param_label", None):
            self.plot_rsp.removeItem(self._param_label)
        self._param_label = None

        if not all(k in cell and cell.get(k) not in ["", "–"] for k in ("mean_rmp", "mean_tau", "mean_rm", "mean_cm")):
            return

        txt = (f"RMP: {cell.get('mean_rmp', '–')} mV    "
               f"Tau: {cell.get('mean_tau', '–')} ms    "
               f"Rm: {cell.get('mean_rm', '–')} MΩ    "
               f"Cm: {cell.get('mean_cm', '–')} pF")

        lbl = pg.TextItem(txt, anchor=(0, 1), color=(0, 0, 0))
        self.plot_rsp.addItem(lbl)
        self._param_label = lbl

        (xmin, xmax), (ymin, ymax) = self.plot_rsp.viewRange()
        lbl.setPos(xmin + 0.03 * (xmax - xmin),
                   ymin + 0.03 * (ymax - ymin))

    def _on_row_selected(self, *args):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        cell = self._cells[row]

        # reset protocol index only when row actually changes
        if args:
            self._current_cell_index_display = 0

        # clear prior plots & state
        for p in (self.plot_cmd, self.plot_rsp):
            p.clear()
        self.legend.clear()
        self._curve_pairs.clear()
        self._selected_curve = None
        self._resp2cmd.clear()

        # placeholder in command plot if checked
        if self.chk_show_cmd.isChecked():
            self.plot_cmd.addItem(pg.TextItem(
                html="<span style='color:blue'>Select a sweep to see corresponding command</span>"
            ))
        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())

        # Which “cell index” to load?
        ids = cell["cell_ids"]
        if not ids:
            self.plot_rsp.addItem(pg.TextItem(
                html="<span style='color:red'>No Current Protocol indices for this UID</span>"
            ))
            if self.chk_show_cmd.isChecked():
                self.plot_cmd.clear()
                self.plot_cmd.addItem(pg.TextItem(
                    html="<span style='color:blue'>No sweeps for this UID</span>"
                ))
            return

        cur_id = ids[self._current_cell_index_display]

        # load raw traces: {current_pA: (t_s, cmd_pA, rsp_V)}
        raw_map = load_current_traces(cell["src_dir"], [cur_id]).get(cur_id, {})
        if not raw_map:
            self.plot_rsp.addItem(pg.TextItem(
                html="<span style='color:red'>No traces found for index %s</span>" % cur_id
            ))
            return

        # progress dialog
        dlg = QtWidgets.QProgressDialog(
            f"Loading sweeps for index {cur_id}…", None, 0, 0, self
        )
        dlg.setWindowTitle("Loading Traces")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        try:
            from functools import partial

            # iterate sorted currents for consistent ordering
            for amp in sorted(raw_map.keys()):
                t_s, cmd_pA, rsp_V = raw_map[amp]
                t_ms = t_s * 1e3
                rsp_mV = rsp_V * 1e3

                pen = pg.mkPen("red", width=1)
                # command (hidden until click)
                curve_cmd = self.plot_cmd.plot(
                    t_ms, cmd_pA, pen=pen, clickable=True
                )
                curve_cmd.setVisible(False)

                # response, labeled by actual current
                label = f"{int(amp)} pA"
                curve_rsp = self.plot_rsp.plot(
                    t_ms, rsp_mV, pen=pen, clickable=True, name=label
                )

                # click both to highlight
                curve_rsp.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))
                curve_cmd.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))

                # legend‐toggle (grid layout set in __init__)
                _, lbl = self.legend.items[-1]
                lbl.setAcceptHoverEvents(True)
                def make_toggle(c1=curve_cmd, c2=curve_rsp, legend_lbl=lbl):
                    def _toggle(_):
                        vis = not c2.isVisible()
                        c2.setVisible(vis)
                        legend_lbl.setOpacity(1.0 if vis else 0.3)
                        if self._selected_curve is c2:
                            c1.setVisible(vis and self.chk_show_cmd.isChecked())
                        else:
                            c1.setVisible(False)
                        self._update_plot_opacities()
                    return _toggle
                lbl.mousePressEvent = make_toggle()

                self._curve_pairs.append((curve_cmd, curve_rsp))
                self._resp2cmd[curve_rsp] = curve_cmd

            self.plot_rsp.enableAutoRange()

        finally:
            dlg.close()

        # update UI
        self._show_param_label(cell)
        self._update_protocol_buttons()

    def _update_protocol_buttons(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.btn_previous_protocol.setEnabled(False)
            self.btn_next_protocol.setEnabled(False)
            return

        cell = self._cells[sel[0].row()]
        num_protocols = len(cell["cell_ids"])

        self.btn_previous_protocol.setEnabled(self._current_cell_index_display > 0)
        self.btn_next_protocol.setEnabled(self._current_cell_index_display < num_protocols - 1)

    def _on_next_protocol_clicked(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel: return
        cell = self._cells[sel[0].row()]
        num_protocols = len(cell["cell_ids"])

        if self._current_cell_index_display < num_protocols - 1:
            self._current_cell_index_display += 1
            self._on_row_selected()

    def _on_previous_protocol_clicked(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel: return

        if self._current_cell_index_display > 0:
            self._current_cell_index_display -= 1
            self._on_row_selected()


    def _on_response_clicked(self, rsp_curve: pg.PlotDataItem, _event=None):
        """
        Highlight the sweep whose *response* curve was clicked.
        Show only its paired command signal, respecting the "Show Command" toggle.
        """
        cmd_curve = self._resp2cmd[rsp_curve]

        # Toggle selection: if already selected, deselect. Otherwise, select.
        is_currently_selected = (self._selected_curve is rsp_curve)
        self._selected_curve = None if is_currently_selected else rsp_curve

        # Always update response plot opacities regardless of command toggle
        self._update_plot_opacities()

        if self.chk_show_cmd.isChecked():
            # If "Show Command" is checked, manage the command plot content
            self.plot_cmd.clear() # Clear existing command traces/messages

            if self._selected_curve: # If a trace is now selected
                self.plot_cmd.addItem(cmd_curve)
                cmd_curve.setVisible(True)
                cmd_curve.setPen("red", width=3)
                cmd_curve.setOpacity(1.0)
                self.plot_cmd.enableAutoRange()
            else: # No trace selected after toggle (i.e., user deselected)
                self.plot_cmd.addItem(pg.TextItem(html="<span style='color:blue'>Select a sweep to see corresponding command</span>"))
        # If self.chk_show_cmd is off, plot_cmd remains hidden, and its content is not managed here.


    def _update_plot_opacities(self):
        """Applies opacity and pen changes based on selected curve to response plots.
           Ensures unselected command curves are hidden unless selected and toggle is on."""
        for r_curve, c_curve in self._resp2cmd.items():
            visible_rsp_via_legend = r_curve.isVisible() # Check if legend toggle hides it

            if r_curve is self._selected_curve:
                opacity_rsp, width_rsp = 1.0, 3
                # If this is the selected curve, and show_cmd is on, make its command visible
                c_curve.setVisible(self.chk_show_cmd.isChecked() and visible_rsp_via_legend)
                if self.chk_show_cmd.isChecked() and visible_rsp_via_legend:
                     c_curve.setPen("red", width=3)
                     c_curve.setOpacity(1.0) # Ensure selected command is opaque
            elif visible_rsp_via_legend: # Other response curves that are visible via legend
                opacity_rsp, width_rsp = 0.25, 1
                c_curve.setVisible(False) # Other command curves are always hidden
            else: # Response curve hidden via legend
                opacity_rsp, width_rsp = 0.05, 1
                c_curve.setVisible(False) # Command curve also hidden

            r_curve.setOpacity(opacity_rsp)
            r_curve.setPen("red", width=width_rsp)


    # ───────────────────────────────────────────────────────── analysis button
    def _on_analyze(self):
        """
        For **every sweep** across all loaded cells:
        • compute passive params, firing rate, and spike metrics
        • append one row per sweep to self._sweep_df
        """
        rows = []
        total_cells = len(self._cells)
        dlg = QtWidgets.QProgressDialog("Analyzing current‑clamp sweeps…",
                                        None, 0, total_cells, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        for idx, cell in enumerate(self._cells, 1):
            dlg.setLabelText(f"Cell {idx} / {total_cells}")
            QtWidgets.QApplication.processEvents()

            # load {cell_id: {amp_pA: (t, cmd_pA, rsp_V)}}
            traces_map = load_current_traces(cell["src_dir"], cell["cell_ids"])
            for cell_id, amp_dict in traces_map.items():
                for amp_pA, (t, cmd_pA, rsp_V) in amp_dict.items():
                    try:
                        # ---------------- passive params ----------------
                        cmd_V   = cmd_pA / self._cprot.clamp_gain
                        rsp_mV  = rsp_V * 1e3
                        pas = self._cprot.passive_params(t, cmd_V, rsp_mV)

                        rmp  = pas["resting_potential_mV"]
                        tau  = pas["membrane_tau_ms"]
                        rm   = pas["input_resistance_MOhm"]*1000  # GΩ → MΩ
                        cm   = pas["membrane_capacitance_pF"] 
                    except Exception:
                        rmp = tau = rm = cm = np.nan      # keep sweep anyway

                    # ---------------- firing rate ----------------------
                    fi = self._cprot.firing_curve([(t, cmd_V, rsp_mV)])
                    fr = float(fi["mean_firing_frequency_Hz"].iloc[0]) \
                         if not fi.empty else np.nan

                    # ---------------- spike metrics -------------------
                    spk = self._cprot.spike_metrics([(t, cmd_V, rsp_mV)])
                    if spk.empty:
                        peak = hwdt = thr = dvdt = np.nan
                    else:
                        peak = spk["peak_mV"].mean()
                        hwdt = spk["half_width_ms"].mean()
                        thr  = spk["threshold_mV"].mean()
                        dvdt = spk["dvdt_max_mV_per_ms"].mean() * 1_000  # → mV/s

                    rows.append({
                        "Cell(UniqueID)"         : cell["unique_id"],
                        "Index"                  : cell_id,
                        "coordinates"            : cell["coord"],
                        "Source Dir"             : cell["src_dir"].name,
                        "Group Label"            : cell["group_label"],
                        "Injected current (pA)"  : amp_pA,
                        "RMP mV"                 : rmp,
                        "Tau ms"                 : tau,
                        "Rm (Mohms)"             : rm,
                        "Cm (pF)"                : cm,
                        "Firing rate (Hz)"       : fr,
                        "mean AP peak mV"        : peak,
                        "mean AP hwdt(ms)"       : hwdt,
                        "threshold"              : thr,
                        "dV/dt max (mV/s)"       : dvdt,
                    })

            dlg.setValue(idx)
            if dlg.wasCanceled():
                break
        dlg.close()

        # build the master per‑sweep DataFrame
        self._sweep_df = pd.DataFrame(rows, columns=_SWEEP_COLS)

        QtWidgets.QMessageBox.information(
            self, "Current‑Clamp Analysis",
            f"Analysis complete – {len(self._sweep_df)} sweeps processed."
        )


    def _analyze_cell(self, traces: dict):
        stats = {cid: "–" for cid in (
            "mean_rmp", "mean_tau", "mean_rm", "mean_cm",
            "rmp_list", "tau_list", "rm_list", "cm_list")}
        stats["has_cc"] = "✓" if traces else "–"

        if not traces:
            return stats

        return stats

    def _update_table_row(self, row: int, stats: dict):
        for cid, val in stats.items():
            if cid not in self._col_idx:
                continue
            item = QtWidgets.QTableWidgetItem(val)
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self._col_idx[cid], item)

    # ─────────────────────────────────────────────────────── export button
    def _save_full_csv(self):
        if self._sweep_df.empty:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to save",
                "Run the analysis first – no per‑sweep data available.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Per‑Sweep Analysis as CSV", "",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        self._sweep_df.to_csv(path, index=False)
