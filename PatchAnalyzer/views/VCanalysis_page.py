# PatchAnalyzer/views/VCanalysis_page.py
from __future__ import annotations

from functools import partial
from pathlib import Path
import hashlib
import numbers
import re

import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..models.data_loader import load_voltage_traces_for_indices
from ..utils.ephys_analyzer import VprotAnalyzer


_COL_IDS = [
    "unique_id", "indices", "src_dir", "group_label",
    "timestamp", "voltage_hold_mV",
    "mean_ra", "sd_ra", "mean_rm", "sd_rm", "mean_cm", "sd_cm",
    "ra_list", "rm_list", "cm_list",
    "has_vprot",
]

_COL_HEADERS = [
    "UID", "Indices", "Source Dir", "Group Label",
    "Timestamp", "Voltage Hold (mV)",
    "Mean Ra (MΩ)", "SD Ra (MΩ)",
    "Mean Rm (MΩ)", "SD Rm (MΩ)",
    "Mean Cm (pF)", "SD Cm (pF)",
    "Ra list (MΩ)", "Rm list (MΩ)", "Cm list (pF)",
    "Has Vprot",
]

_STAT_COL_IDS = (
    "mean_ra", "sd_ra", "mean_rm", "sd_rm", "mean_cm", "sd_cm",
    "ra_list", "rm_list", "cm_list", "has_vprot",
)

_COORD_COLS = ("stage_x", "stage_y", "stage_z")
_ID_RE = re.compile(r"cell[_\-]?(\d+)", re.IGNORECASE)
_VC_TRACE_RE = re.compile(r"^(?:VoltageProtocol|MembraneTest)_(\d+)", re.IGNORECASE)


def _cell_id(img_name: str) -> int | None:
    match = _ID_RE.search(img_name)
    return int(match.group(1)) if match else None


def _trace_index_from_name(file_name: str) -> int | None:
    match = _VC_TRACE_RE.match(Path(file_name).stem)
    return int(match.group(1)) if match else None


def _format_meta_value(val) -> str:
    if isinstance(val, str):
        text = val.strip()
        if not text or text.lower() == "nan":
            return ""
        return text
    if pd.isna(val):
        return ""
    if isinstance(val, numbers.Real):
        value = float(val)
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return str(val)


def _collapse_column(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    seen: list[str] = []
    for val in df[column]:
        text = _format_meta_value(val)
        if text and text not in seen:
            seen.append(text)
    return ", ".join(seen)


class VCAnalysisPage(QtWidgets.QWidget):
    """Left-hand cell table, command/response plots, and bottom controls."""

    back_requested = QtCore.pyqtSignal()
    analyze_requested = QtCore.pyqtSignal(pd.DataFrame)
    continue_requested = QtCore.pyqtSignal()

    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.meta_df = meta_df.reset_index(drop=True)
        for col in ("timestamp", "voltage_hold_mV"):
            if col not in self.meta_df.columns:
                self.meta_df[col] = ""

        self._vprot = VprotAnalyzer(step_mV=10.0)
        self._cells: list[dict] = []
        self._uid2cell: dict[int, dict] = {}
        self._omitted_vc_files: dict[tuple[str, object, object, object], set[str]] = {}
        self._param_label: pg.TextItem | None = None
        self._param_conn = None

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
        self.plot_cmd.setLabel("left", "Command (mV)")
        self.plot_cmd.setLabel("bottom", "Time (ms)")
        right_lay.addWidget(self.plot_cmd, 1)

        self.plot_rsp = pg.PlotWidget(background="w")
        self.plot_rsp.setLabel("left", "Response (pA)")
        self.plot_rsp.setLabel("bottom", "Time (ms)")
        self.legend = self.plot_rsp.addLegend(offset=(-110, 30))
        right_lay.addWidget(self.plot_rsp, 2)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(1, 1)

        bar = QtWidgets.QHBoxLayout()
        self.btn_save_csv = QtWidgets.QPushButton("Save Full CSV…")
        self.btn_save_csv.clicked.connect(self._save_full_csv)
        bar.addWidget(self.btn_save_csv)

        self.btn_analyze = QtWidgets.QPushButton("Analyze ▶")
        self.btn_analyze.clicked.connect(self._on_analyze)
        bar.addWidget(self.btn_analyze)

        self.lbl_protocol_file = QtWidgets.QLabel("Protocol File:")
        bar.addWidget(self.lbl_protocol_file)

        self.cmb_protocol_file = QtWidgets.QComboBox()
        self.cmb_protocol_file.setEnabled(False)
        self.cmb_protocol_file.setMinimumContentsLength(24)
        self.cmb_protocol_file.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        bar.addWidget(self.cmb_protocol_file)

        self.btn_omit_protocol = QtWidgets.QPushButton("Omit Protocol")
        self.btn_omit_protocol.setEnabled(False)
        self.btn_omit_protocol.clicked.connect(self._on_omit_protocol)
        bar.addWidget(self.btn_omit_protocol)

        bar.addStretch(1)

        self.chk_show_cmd = QtWidgets.QCheckBox("Show Command")
        self.chk_show_cmd.setChecked(True)
        bar.addWidget(self.chk_show_cmd)

        self.btn_back = QtWidgets.QPushButton("← Back")
        self.btn_back.clicked.connect(self.back_requested)
        bar.addWidget(self.btn_back)

        self.btn_continue = QtWidgets.QPushButton("Continue ▶")
        self.btn_continue.clicked.connect(self.continue_requested)
        bar.addWidget(self.btn_continue)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(splitter, 1)
        root.addLayout(bar)

        self.chk_show_cmd.toggled.connect(self.plot_cmd.setVisible)

        self._curve_pairs: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []
        self._selected_pair: tuple[pg.PlotDataItem, pg.PlotDataItem] | None = None
        self._selected_curve: pg.PlotDataItem | None = None
        self._resp2cmd: dict[pg.PlotDataItem, pg.PlotDataItem] = {}

        self._rebuild_cells()
        self._populate_table()
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)

        if self.table.rowCount():
            self.table.selectRow(0)
            self._on_row_selected()
        else:
            self._clear_selection_ui()

    def _make_cell_key(
        self,
        src_dir: Path,
        coord: tuple[object, object, object],
    ) -> tuple[str, object, object, object]:
        return (str(src_dir), coord[0], coord[1], coord[2])

    def _rebuild_cells(self) -> None:
        old_cells = {cell["cell_key"]: cell for cell in getattr(self, "_cells", [])}
        self._cells = []

        if self.meta_df.empty:
            self._uid2cell = {}
            return

        for uid, (coord, sub) in enumerate(self.meta_df.groupby(list(_COORD_COLS)), start=1):
            src_dir_path = Path(sub["src_dir"].iloc[0])
            cell_ids = [_cell_id(img) for img in sub["image"] if _cell_id(img) is not None]
            cell_key = self._make_cell_key(src_dir_path, coord)
            cell = dict(
                unique_id=uid,
                cell_key=cell_key,
                coord=coord,
                src_dir=src_dir_path,
                group_label=sub["group_label"].iloc[0],
                cell_ids=sorted(set(cell_ids)),
                timestamp=_collapse_column(sub, "timestamp"),
                voltage_hold_mV=_collapse_column(sub, "voltage_hold_mV"),
            )

            old_cell = old_cells.get(cell_key)
            if old_cell:
                for key in _STAT_COL_IDS:
                    if key in old_cell:
                        cell[key] = old_cell[key]

            self._cells.append(cell)

        self._uid2cell = {cell["unique_id"]: cell for cell in self._cells}

    def _populate_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setColumnCount(len(_COL_IDS))
        self.table.setHorizontalHeaderLabels(_COL_HEADERS)
        self.table.setRowCount(len(self._cells))
        self._col_idx = {cid: i for i, cid in enumerate(_COL_IDS)}

        for row, cell in enumerate(self._cells):
            init_vals = {
                "unique_id": str(cell["unique_id"]),
                "indices": ", ".join(map(str, cell["cell_ids"])),
                "src_dir": cell["src_dir"].name,
                "group_label": cell["group_label"],
                "timestamp": cell.get("timestamp", ""),
                "voltage_hold_mV": cell.get("voltage_hold_mV", ""),
                "mean_ra": cell.get("mean_ra", ""),
                "sd_ra": cell.get("sd_ra", ""),
                "mean_rm": cell.get("mean_rm", ""),
                "sd_rm": cell.get("sd_rm", ""),
                "mean_cm": cell.get("mean_cm", ""),
                "sd_cm": cell.get("sd_cm", ""),
                "ra_list": cell.get("ra_list", ""),
                "rm_list": cell.get("rm_list", ""),
                "cm_list": cell.get("cm_list", ""),
                "has_vprot": cell.get("has_vprot", ""),
            }

            for cid, text in init_vals.items():
                item = QtWidgets.QTableWidgetItem(str(text))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                if cid == "group_label" and text:
                    hue = int(hashlib.md5(str(text).encode()).hexdigest(), 16) % 360
                    item.setBackground(QtGui.QColor.fromHsl(hue, 160, 200))
                    item.setForeground(QtGui.QColor("black"))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, self._col_idx[cid], item)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def _table_row_for_uid(self, uid: int) -> int | None:
        uid_text = str(uid)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self._col_idx["unique_id"])
            if item is not None and item.text() == uid_text:
                return row
        return None

    def _table_row_for_cell_key(
        self,
        cell_key: tuple[str, object, object, object],
    ) -> int | None:
        for cell in self._cells:
            if cell["cell_key"] == cell_key:
                return self._table_row_for_uid(cell["unique_id"])
        return None

    def _current_cell(self) -> tuple[int | None, dict | None]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None, None

        row_in_view = sel[0].row()
        uid_item = self.table.item(row_in_view, self._col_idx["unique_id"])
        if uid_item is None:
            return None, None

        uid = int(uid_item.text())
        return row_in_view, self._uid2cell.get(uid)

    def _clear_selection_ui(self) -> None:
        for plot in (self.plot_cmd, self.plot_rsp):
            plot.clear()
        self.legend.clear()
        self._curve_pairs.clear()
        self._selected_pair = None
        self._selected_curve = None
        self._resp2cmd.clear()
        self._param_label = None
        self._set_protocol_file_options([])
        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())

    def _set_protocol_file_options(
        self,
        names: list[str],
        preferred_name: str | None = None,
    ) -> None:
        self.cmb_protocol_file.blockSignals(True)
        self.cmb_protocol_file.clear()
        for name in names:
            self.cmb_protocol_file.addItem(name, name)

        if names:
            current_name = preferred_name if preferred_name in names else names[0]
            self.cmb_protocol_file.setCurrentIndex(names.index(current_name))

        enabled = bool(names)
        self.cmb_protocol_file.setEnabled(enabled)
        self.btn_omit_protocol.setEnabled(enabled)
        self.cmb_protocol_file.blockSignals(False)

    def _show_param_label(self, cell: dict) -> None:
        if getattr(self, "_param_label", None):
            self.plot_rsp.removeItem(self._param_label)
        self._param_label = None

        if not all(key in cell for key in ("mean_ra", "mean_rm", "mean_cm")):
            return

        txt = (
            f"Ra: {cell['mean_ra']} MΩ    "
            f"Rm: {cell['mean_rm']} MΩ    "
            f"Cm: {cell['mean_cm']} pF"
        )

        label = pg.TextItem(txt, anchor=(0, 1), color=(0, 0, 0))
        self.plot_rsp.addItem(label)
        self._param_label = label

        (xmin, xmax), (ymin, ymax) = self.plot_rsp.viewRange()
        label.setPos(xmin + 0.03 * (xmax - xmin), ymin + 0.03 * (ymax - ymin))

    def _all_traces_for_cell(self, cell: dict) -> dict[str, tuple[object, object, object]]:
        traces = load_voltage_traces_for_indices(cell["src_dir"], cell["cell_ids"])
        return {name: traces[name] for name in sorted(traces)}

    def _active_traces_for_cell(self, cell: dict) -> dict[str, tuple[object, object, object]]:
        omitted = self._omitted_vc_files.get(cell["cell_key"], set())
        traces = self._all_traces_for_cell(cell)
        return {name: trace for name, trace in traces.items() if name not in omitted}

    def _on_row_selected(self, *_args) -> None:
        _, cell = self._current_cell()
        if cell is None:
            self._clear_selection_ui()
            return

        self._clear_selection_ui()

        if self.chk_show_cmd.isChecked():
            self.plot_cmd.addItem(
                pg.TextItem(
                    html="<span style='color:blue'>Select a sweep to see corresponding command</span>"
                )
            )
        self.plot_cmd.setVisible(self.chk_show_cmd.isChecked())

        traces = self._active_traces_for_cell(cell)
        trace_names = list(traces)
        self._set_protocol_file_options(trace_names)

        if not traces:
            self.plot_rsp.addItem(
                pg.TextItem(html="<span style='color:red'>No traces found</span>")
            )
            self._show_param_label(cell)
            return

        for fname, (t, cmd, rsp) in traces.items():
            t_ms = t * 1e3
            cmd_mV = cmd * 1e3
            rsp_pA = rsp * 1e12

            pen = pg.mkPen("red", width=1)
            curve_cmd = self.plot_cmd.plot(t_ms, cmd_mV, pen=pen, clickable=True)
            curve_rsp = self.plot_rsp.plot(
                t_ms, rsp_pA, pen=pen, clickable=True, name=fname
            )

            curve_rsp.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))
            curve_cmd.sigClicked.connect(partial(self._on_response_clicked, curve_rsp))

            label_item = self.legend.items[-1][1]
            label_item.setAcceptHoverEvents(True)

            def _make_toggle(c1=curve_cmd, c2=curve_rsp, lbl=label_item):
                def _toggle(_event):
                    visible = not c1.isVisible()
                    for curve in (c1, c2):
                        curve.setVisible(visible)
                    lbl.setOpacity(1.0 if visible else 0.3)
                return _toggle

            label_item.mousePressEvent = _make_toggle()

            self._curve_pairs.append((curve_cmd, curve_rsp))
            self._resp2cmd[curve_rsp] = curve_cmd

        self.plot_cmd.enableAutoRange()
        self.plot_rsp.enableAutoRange()
        self._show_param_label(cell)

    def _on_response_clicked(self, rsp_curve: pg.PlotDataItem, _event=None) -> None:
        cmd_curve = self._resp2cmd[rsp_curve]
        self._selected_curve = (
            rsp_curve if self._selected_curve is not rsp_curve else None
        )

        for r_curve, c_curve in self._resp2cmd.items():
            visible = r_curve.isVisible() or c_curve.isVisible()
            selected = (r_curve is self._selected_curve) and visible

            if selected:
                opacity, width = 1.0, 3
            elif visible:
                opacity, width = 0.25, 1
            else:
                opacity, width = 0.05, 1

            for curve in (r_curve, c_curve):
                curve.setOpacity(opacity)
                curve.setPen("red", width=width)

    def _on_analyze(self) -> None:
        total = len(self._cells)
        dlg = QtWidgets.QProgressDialog(
            "Analyzing passive parameters…", None, 0, total, self
        )
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        for idx, cell in enumerate(self._cells, start=1):
            dlg.setLabelText(f"Cell {idx} / {total}")
            QtWidgets.QApplication.processEvents()

            stats = self._analyze_cell(self._active_traces_for_cell(cell))
            cell.update(stats)
            self._update_table_row(cell["unique_id"], stats)

            dlg.setValue(idx)
            if dlg.wasCanceled():
                break

        dlg.close()

        if self.table.currentRow() >= 0:
            self._on_row_selected()

        self.analyze_requested.emit(self.meta_df)

    def _analyze_cell(self, traces: dict) -> dict[str, str]:
        stats = {
            cid: "–"
            for cid in (
                "mean_ra", "sd_ra", "mean_rm", "sd_rm", "mean_cm", "sd_cm",
                "ra_list", "rm_list", "cm_list",
            )
        }
        stats["has_vprot"] = "✓" if traces else "–"

        if not traces:
            return stats

        _, fits = self._vprot.fit_cell(traces, aggregate="mean", return_all=True)
        ok = [fit for fit in fits if all(value is not None for value in fit)]
        if not ok:
            return stats

        import numpy as np

        ra, rm, cm = zip(*ok)

        def _fmt(value: float) -> str:
            return f"{value:.1f}"

        def _sd(values) -> float:
            return np.std(values, ddof=1) if len(values) > 1 else 0.0

        stats.update(
            mean_ra=_fmt(np.mean(ra)),
            sd_ra=_fmt(_sd(ra)),
            mean_rm=_fmt(np.mean(rm)),
            sd_rm=_fmt(_sd(rm)),
            mean_cm=_fmt(np.mean(cm)),
            sd_cm=_fmt(_sd(cm)),
            ra_list=", ".join(_fmt(value) for value in ra),
            rm_list=", ".join(_fmt(value) for value in rm),
            cm_list=", ".join(_fmt(value) for value in cm),
        )
        return stats

    def _update_table_row(self, uid: int, stats: dict[str, str]) -> None:
        row = self._table_row_for_uid(uid)
        if row is None:
            return

        for cid, val in stats.items():
            if cid not in self._col_idx:
                continue
            item = QtWidgets.QTableWidgetItem(str(val))
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self._col_idx[cid], item)

    def _prune_meta_rows_for_index(self, cell: dict, protocol_index: int) -> None:
        if "index" not in self.meta_df.columns:
            return

        index_values = pd.to_numeric(self.meta_df["index"], errors="coerce")
        mask = (
            (self.meta_df["src_dir"] == cell["src_dir"])
            & (self.meta_df["stage_x"] == cell["coord"][0])
            & (self.meta_df["stage_y"] == cell["coord"][1])
            & (self.meta_df["stage_z"] == cell["coord"][2])
            & index_values.eq(protocol_index)
        )
        if mask.any():
            self.meta_df = self.meta_df.loc[~mask].reset_index(drop=True)

    def _rebuild_after_omission(
        self,
        preferred_cell_key: tuple[str, object, object, object],
        fallback_row: int,
    ) -> None:
        self._rebuild_cells()
        self._populate_table()

        if not self._cells:
            self._clear_selection_ui()
            return

        row_to_select = self._table_row_for_cell_key(preferred_cell_key)
        if row_to_select is None:
            row_to_select = min(max(fallback_row, 0), self.table.rowCount() - 1)

        self.table.selectRow(row_to_select)
        self._on_row_selected()

        _, cell = self._current_cell()
        if cell is None or cell["cell_key"] != preferred_cell_key:
            return
        if "has_vprot" not in cell:
            return

        stats = self._analyze_cell(self._active_traces_for_cell(cell))
        cell.update(stats)
        self._update_table_row(cell["unique_id"], stats)
        self._show_param_label(cell)

    def _on_omit_protocol(self) -> None:
        row_in_view, cell = self._current_cell()
        if cell is None:
            return

        file_name = self.cmb_protocol_file.currentData()
        if not file_name:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Omit Protocol",
            f"Omit protocol file {file_name} from further analysis?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        omitted = self._omitted_vc_files.setdefault(cell["cell_key"], set())
        omitted.add(file_name)

        protocol_index = _trace_index_from_name(file_name)
        if protocol_index is not None:
            remaining_same_index = any(
                _trace_index_from_name(name) == protocol_index
                for name in self._active_traces_for_cell(cell)
            )
            if not remaining_same_index:
                self._prune_meta_rows_for_index(cell, protocol_index)

        self._rebuild_after_omission(cell["cell_key"], row_in_view or 0)

    def _save_full_csv(self) -> None:
        rows = []
        for cell in self._cells:
            rows.append(
                {
                    "UID": cell["unique_id"],
                    "indices": ", ".join(map(str, cell["cell_ids"])),
                    "stage_x": cell["coord"][0],
                    "stage_y": cell["coord"][1],
                    "stage_z": cell["coord"][2],
                    "src_dir": cell["src_dir"].name,
                    "group_label": cell["group_label"],
                    "timestamp": cell.get("timestamp", ""),
                    "voltage_hold_mV": cell.get("voltage_hold_mV", ""),
                    "mean_Ra_MOhm": cell.get("mean_ra", "–"),
                    "SD_Ra_MOhm": cell.get("sd_ra", "–"),
                    "mean_Rm_MOhm": cell.get("mean_rm", "–"),
                    "SD_Rm_MOhm": cell.get("sd_rm", "–"),
                    "mean_Cm_pF": cell.get("mean_cm", "–"),
                    "SD_Cm_pF": cell.get("sd_cm", "–"),
                    "Ra_list_MOhm": cell.get("ra_list", ""),
                    "Rm_list_MOhm": cell.get("rm_list", ""),
                    "Cm_list_pF": cell.get("cm_list", ""),
                    "has_Vprot": cell.get("has_vprot", "–"),
                }
            )

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        pd.DataFrame(rows).to_csv(path, index=False)
