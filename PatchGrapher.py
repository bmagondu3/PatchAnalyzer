from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5 import QtCore, QtGui, QtWidgets

from PatchAnalyzer.utils import ephys_grapher3 as grapher


PLOT_ORDER = list(grapher.ALL_PARAMS) + ["FiringCurve"]
DEFAULT_ACCESS_RESISTANCE_MAX = 30.0


class PatchGrapherWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._prepared_data: dict[str, Any] | None = None
        self._prepared_key: tuple[str, str, str, float] | None = None
        self._group_controls: dict[str, dict[str, QtWidgets.QWidget]] = {}
        self._current_canvas: FigureCanvasQTAgg | None = None
        self._current_toolbar: NavigationToolbar2QT | None = None
        self._current_figure = None
        self._updating_cell_table = False

        self.setWindowTitle("Patch Grapher")
        self.resize(1450, 900)
        self._build_ui()
        self._apply_defaults()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)

        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        controls_scroll = QtWidgets.QScrollArea(self)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_panel = QtWidgets.QWidget(self)
        controls_scroll.setWidget(controls_panel)
        splitter.addWidget(controls_scroll)

        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._build_files_group())
        controls_layout.addWidget(self._build_plot_options_group())
        controls_layout.addWidget(self._build_plot_category_group())
        controls_layout.addWidget(self._build_group_category_group())
        controls_layout.addWidget(self._build_cell_group())
        controls_layout.addWidget(self._build_action_group())
        controls_layout.addStretch(1)

        preview_panel = QtWidgets.QWidget(self)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 980])

        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        preview_layout.setSpacing(8)

        preview_header = QtWidgets.QHBoxLayout()
        preview_layout.addLayout(preview_header)
        preview_header.addWidget(QtWidgets.QLabel("Preview Plot:", self))

        self.preview_combo = QtWidgets.QComboBox(self)
        self.preview_combo.currentIndexChanged.connect(self._refresh_preview)
        preview_header.addWidget(self.preview_combo, 1)

        self.canvas_host = QtWidgets.QWidget(self)
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_host)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(4)
        preview_layout.addWidget(self.canvas_host, 1)

        self._placeholder = QtWidgets.QLabel(
            "Select CC and VC summary CSVs, then click Load Data to preview backend-generated graphs.",
            self,
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self.canvas_layout.addWidget(self._placeholder, 1)

        self.stats_box = QtWidgets.QPlainTextEdit(self)
        self.stats_box.setReadOnly(True)
        self.stats_box.setMaximumHeight(180)
        preview_layout.addWidget(self.stats_box)

        self.statusBar().showMessage("Choose the CC and VC summary CSVs to begin.")

    def _build_files_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Files", self)
        layout = QtWidgets.QGridLayout(box)
        layout.setColumnStretch(1, 1)

        self.cc_path_edit = QtWidgets.QLineEdit(self)
        self.vc_path_edit = QtWidgets.QLineEdit(self)
        self.save_dir_edit = QtWidgets.QLineEdit(self)

        cc_btn = QtWidgets.QPushButton("Browse...", self)
        cc_btn.clicked.connect(lambda: self._choose_csv(self.cc_path_edit, "Select current-clamp CSV"))
        vc_btn = QtWidgets.QPushButton("Browse...", self)
        vc_btn.clicked.connect(lambda: self._choose_csv(self.vc_path_edit, "Select voltage-clamp CSV"))
        save_btn = QtWidgets.QPushButton("Folder...", self)
        save_btn.clicked.connect(self._choose_save_dir)

        layout.addWidget(QtWidgets.QLabel("CC CSV", self), 0, 0)
        layout.addWidget(self.cc_path_edit, 0, 1)
        layout.addWidget(cc_btn, 0, 2)
        layout.addWidget(QtWidgets.QLabel("VC CSV", self), 1, 0)
        layout.addWidget(self.vc_path_edit, 1, 1)
        layout.addWidget(vc_btn, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Save Folder", self), 2, 0)
        layout.addWidget(self.save_dir_edit, 2, 1)
        layout.addWidget(save_btn, 2, 2)
        return box

    def _build_plot_options_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Plot Options", self)
        form = QtWidgets.QFormLayout(box)

        self.cm_source_combo = QtWidgets.QComboBox(self)
        self.cm_source_combo.addItems(["VOLTAGE", "CURRENT"])

        self.ra_max_spin = self._make_spinbox(1.0, 1000.0, 1.0, DEFAULT_ACCESS_RESISTANCE_MAX)
        self.ra_max_spin.setSuffix(" MOhm")

        self.firing_x_min = self._make_spinbox(-50.0, 500.0, 0.5, 0.0)
        self.firing_x_max = self._make_spinbox(-50.0, 500.0, 0.5, 13.0)
        self.firing_y_min = self._make_spinbox(-10.0, 1000.0, 1.0, 0.0)
        self.firing_y_max = self._make_spinbox(-10.0, 1000.0, 1.0, 100.0)

        x_bounds = QtWidgets.QHBoxLayout()
        x_bounds.addWidget(self.firing_x_min)
        x_bounds.addWidget(QtWidgets.QLabel("to", self))
        x_bounds.addWidget(self.firing_x_max)

        y_bounds = QtWidgets.QHBoxLayout()
        y_bounds.addWidget(self.firing_y_min)
        y_bounds.addWidget(QtWidgets.QLabel("to", self))
        y_bounds.addWidget(self.firing_y_max)

        form.addRow("Capacitance Source", self.cm_source_combo)
        form.addRow("Max Access Resistance", self.ra_max_spin)
        form.addRow("Firing X Bounds", x_bounds)
        form.addRow("Firing Y Bounds", y_bounds)
        return box

    def _build_plot_category_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Plot Categories", self)
        layout = QtWidgets.QVBoxLayout(box)

        button_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select All", self)
        select_all.clicked.connect(lambda: self._set_all_plot_checks(True))
        clear_all = QtWidgets.QPushButton("Clear All", self)
        clear_all.clicked.connect(lambda: self._set_all_plot_checks(False))
        button_row.addWidget(select_all)
        button_row.addWidget(clear_all)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self.plot_checks: dict[str, QtWidgets.QCheckBox] = {}
        for index, key in enumerate(PLOT_ORDER):
            checkbox = QtWidgets.QCheckBox(self._plot_label(key), self)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._sync_preview_choices)
            self.plot_checks[key] = checkbox
            grid.addWidget(checkbox, index // 2, index % 2)

        return box
    def _build_group_category_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Group Categories", self)
        layout = QtWidgets.QVBoxLayout(box)

        help_text = QtWidgets.QLabel(
            "Detected CSV group labels appear here after loading. Uncheck a category to exclude it, or change its color before previewing or saving.",
            self,
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.group_area = QtWidgets.QWidget(self)
        self.group_layout = QtWidgets.QVBoxLayout(self.group_area)
        self.group_layout.setContentsMargins(0, 0, 0, 0)
        self.group_layout.setSpacing(6)
        layout.addWidget(self.group_area)

        self.group_placeholder = QtWidgets.QLabel("No categories loaded yet.", self)
        self.group_layout.addWidget(self.group_placeholder)
        self.group_layout.addStretch(1)
        return box

    def _build_cell_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Individual Cells", self)
        layout = QtWidgets.QVBoxLayout(box)

        help_text = QtWidgets.QLabel(
            "Cells listed here have already passed the access-resistance gate. Uncheck a UID to exclude it from every preview and saved graph.",
            self,
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        button_row = QtWidgets.QHBoxLayout()
        enable_all = QtWidgets.QPushButton("Enable All", self)
        enable_all.clicked.connect(lambda: self._set_all_cell_checks(True))
        disable_all = QtWidgets.QPushButton("Disable All", self)
        disable_all.clicked.connect(lambda: self._set_all_cell_checks(False))
        self.cell_summary_label = QtWidgets.QLabel("No cells loaded yet.", self)
        button_row.addWidget(enable_all)
        button_row.addWidget(disable_all)
        button_row.addStretch(1)
        button_row.addWidget(self.cell_summary_label)
        layout.addLayout(button_row)

        self.cell_table = QtWidgets.QTableWidget(self)
        self.cell_table.setColumnCount(4)
        self.cell_table.setHorizontalHeaderLabels(["Use", "UID", "Group", "Ra (MOhm)"])
        self.cell_table.verticalHeader().setVisible(False)
        self.cell_table.setAlternatingRowColors(True)
        self.cell_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cell_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.cell_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cell_table.setSortingEnabled(False)
        self.cell_table.setMinimumHeight(260)
        header = self.cell_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.cell_table.itemChanged.connect(self._on_cell_item_changed)
        layout.addWidget(self.cell_table)
        return box

    def _build_action_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Actions", self)
        layout = QtWidgets.QHBoxLayout(box)

        load_btn = QtWidgets.QPushButton("Load Data", self)
        load_btn.clicked.connect(self._load_data)
        preview_btn = QtWidgets.QPushButton("Refresh Preview", self)
        preview_btn.clicked.connect(self._refresh_preview)
        save_btn = QtWidgets.QPushButton("Save Graphs", self)
        save_btn.clicked.connect(self._save_graphs)

        layout.addWidget(load_btn)
        layout.addWidget(preview_btn)
        layout.addWidget(save_btn)
        layout.addStretch(1)
        return box

    def _apply_defaults(self) -> None:
        cc_default = Path("Data") / "Forest_HEK_exp" / "corrected" / "CC_HEK_EXP.csv"
        vc_default = Path("Data") / "Forest_HEK_exp" / "corrected" / "VC_HEK_EXP.csv"
        if cc_default.exists():
            self.cc_path_edit.setText(str(cc_default.resolve()))
            self.save_dir_edit.setText(str((cc_default.parent / "PatchGrapher_output").resolve()))
        if vc_default.exists():
            self.vc_path_edit.setText(str(vc_default.resolve()))
        self._sync_preview_choices()

    def _make_spinbox(self, minimum: float, maximum: float, step: float, value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox(self)
        box.setRange(minimum, maximum)
        box.setDecimals(1)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _plot_label(self, key: str) -> str:
        if key == "FiringCurve":
            return "Firing Curve"
        return grapher.PARAM_SPECS[key].title

    def _choose_csv(self, target: QtWidgets.QLineEdit, title: str) -> None:
        start_dir = target.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            title,
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            target.setText(path)
            if not self.save_dir_edit.text().strip():
                self.save_dir_edit.setText(str(Path(path).resolve().parent / "PatchGrapher_output"))

    def _choose_save_dir(self) -> None:
        start_dir = self.save_dir_edit.text().strip() or str(Path.cwd())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if path:
            self.save_dir_edit.setText(path)

    def _set_all_plot_checks(self, checked: bool) -> None:
        for checkbox in self.plot_checks.values():
            checkbox.setChecked(checked)
        self._sync_preview_choices()

    def _set_all_cell_checks(self, checked: bool) -> None:
        if self.cell_table.rowCount() == 0:
            return
        self._updating_cell_table = True
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(self.cell_table.rowCount()):
            item = self.cell_table.item(row, 0)
            if item is not None:
                item.setCheckState(state)
        self._updating_cell_table = False
        self._update_cell_summary()
        self._refresh_preview()

    def _sync_preview_choices(self) -> None:
        current_key = self.preview_combo.currentData()
        selected = [key for key in PLOT_ORDER if self.plot_checks[key].isChecked()]
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        for key in selected:
            self.preview_combo.addItem(self._plot_label(key), key)
        if selected:
            target = current_key if current_key in selected else selected[0]
            index = self.preview_combo.findData(target)
            if index >= 0:
                self.preview_combo.setCurrentIndex(index)
        self.preview_combo.blockSignals(False)

    def _load_data(self) -> None:
        try:
            data = self._get_prepared_data(force_reload=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            self.statusBar().showMessage("Failed to load CSV data.")
            return

        self._sync_preview_choices()
        self.statusBar().showMessage(
            f"Loaded {len(data['groups'])} group categories and {self.cell_table.rowCount()} cells with access resistance <= {self._access_resistance_max():.1f} MOhm."
        )
        if self.preview_combo.count():
            self._refresh_preview()

    def _get_prepared_data(self, force_reload: bool = False) -> dict[str, Any]:
        cc_path = Path(self.cc_path_edit.text().strip()).expanduser()
        vc_path = Path(self.vc_path_edit.text().strip()).expanduser()
        cm_source = self.cm_source_combo.currentText().strip().upper()
        ra_max = self._access_resistance_max()

        if not cc_path.is_file():
            raise FileNotFoundError(f"Current-clamp CSV not found: {cc_path}")
        if not vc_path.is_file():
            raise FileNotFoundError(f"Voltage-clamp CSV not found: {vc_path}")
        if cc_path.suffix.lower() != ".csv" or vc_path.suffix.lower() != ".csv":
            raise ValueError("Both selected files must be CSV files.")

        key = (str(cc_path.resolve()), str(vc_path.resolve()), cm_source, ra_max)
        if force_reload or self._prepared_data is None or self._prepared_key != key:
            cell_means, groups, fi_stats, total_cells, fi_cell = grapher.prepare_cell_means(
                cc_csv=cc_path,
                vc_csv=vc_path,
                cm_source=cm_source,
                cm_range=grapher.CM_RANGE,
                tau_max=grapher.TAU_MAX,
                i_ratio_max=grapher.I_RATIO_MAX,
                bin_step=grapher.BIN_STEP,
                ra_max=ra_max,
            )
            self._prepared_data = {
                "cell_means": cell_means,
                "groups": groups,
                "fi_stats": fi_stats,
                "total_cells": total_cells,
                "fi_cell": fi_cell,
            }
            self._prepared_key = key
            self._populate_group_controls(groups)
            self._populate_cell_table(cell_means)
            if not self.save_dir_edit.text().strip():
                self.save_dir_edit.setText(str(cc_path.resolve().parent / "PatchGrapher_output"))

        return self._prepared_data

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _populate_group_controls(self, groups: list[str]) -> None:
        previous_enabled = {
            group: controls["check"].isChecked()
            for group, controls in self._group_controls.items()
            if "check" in controls
        }
        previous_colors = {
            group: str(controls["button"].property("color") or "")
            for group, controls in self._group_controls.items()
            if "button" in controls
        }

        self._group_controls = {}
        self._clear_layout(self.group_layout)

        if not groups:
            self.group_layout.addWidget(QtWidgets.QLabel("No categories loaded yet.", self))
            self.group_layout.addStretch(1)
            return

        default_colors = grapher._generate_group_colors(len(groups))
        for group, default_color in zip(groups, default_colors):
            row_widget = QtWidgets.QWidget(self)
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            checkbox = QtWidgets.QCheckBox(str(group), self)
            checkbox.setChecked(previous_enabled.get(group, True))
            checkbox.stateChanged.connect(self._refresh_preview)

            color_button = QtWidgets.QPushButton(self)
            color_button.setFixedWidth(120)
            chosen_color = previous_colors.get(group) or str(default_color)
            self._set_color_button(color_button, chosen_color)
            color_button.clicked.connect(lambda _checked=False, g=group: self._pick_group_color(g))

            row_layout.addWidget(checkbox, 1)
            row_layout.addWidget(QtWidgets.QLabel("Color", self))
            row_layout.addWidget(color_button)
            self.group_layout.addWidget(row_widget)
            self._group_controls[group] = {"check": checkbox, "button": color_button}

        self.group_layout.addStretch(1)

    def _populate_cell_table(self, cell_means: pd.DataFrame) -> None:
        previous_enabled = self._cell_enabled_map()
        display = cell_means.loc[:, ["UID", "Group", "Ra"]].drop_duplicates(subset=["UID"]).copy()
        if not display.empty:
            display["UID"] = display["UID"].astype(str)
            display["Group"] = display["Group"].astype(str)
            display = display.sort_values(["Group", "UID"], kind="stable")

        self._updating_cell_table = True
        self.cell_table.clearContents()
        self.cell_table.setRowCount(len(display))

        for row_index, record in enumerate(display.itertuples(index=False)):
            uid = str(record.UID)
            group = str(record.Group)
            ra = record.Ra

            use_item = QtWidgets.QTableWidgetItem()
            use_item.setFlags(
                QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsSelectable
                | QtCore.Qt.ItemIsUserCheckable
            )
            use_item.setCheckState(
                QtCore.Qt.Checked if previous_enabled.get(uid, True) else QtCore.Qt.Unchecked
            )
            use_item.setData(QtCore.Qt.UserRole, uid)
            self.cell_table.setItem(row_index, 0, use_item)

            uid_item = QtWidgets.QTableWidgetItem(uid)
            uid_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.cell_table.setItem(row_index, 1, uid_item)

            group_item = QtWidgets.QTableWidgetItem(group)
            group_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.cell_table.setItem(row_index, 2, group_item)

            ra_text = "NA" if pd.isna(ra) else f"{float(ra):.1f}"
            ra_item = QtWidgets.QTableWidgetItem(ra_text)
            ra_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.cell_table.setItem(row_index, 3, ra_item)

        self._updating_cell_table = False
        self._update_cell_summary()

    def _pick_group_color(self, group: str) -> None:
        controls = self._group_controls.get(group)
        if not controls:
            return
        button = controls["button"]
        current = QtGui.QColor(str(button.property("color") or "#000000"))
        color = QtWidgets.QColorDialog.getColor(current, self, f"Select color for {group}")
        if not color.isValid():
            return
        self._set_color_button(button, color.name().upper())
        self._refresh_preview()

    def _set_color_button(self, button: QtWidgets.QPushButton, color: str) -> None:
        normalized = str(color).strip() or "#000000"
        button.setProperty("color", normalized)
        button.setText(normalized.upper())
        button.setStyleSheet(
            f"QPushButton {{ background-color: {normalized}; color: white; border: 1px solid #666; padding: 4px 8px; }}"
        )

    def _cell_enabled_map(self) -> dict[str, bool]:
        enabled: dict[str, bool] = {}
        for row in range(self.cell_table.rowCount()):
            item = self.cell_table.item(row, 0)
            if item is None:
                continue
            uid = str(item.data(QtCore.Qt.UserRole) or self.cell_table.item(row, 1).text())
            enabled[uid] = item.checkState() == QtCore.Qt.Checked
        return enabled

    def _selected_groups(self) -> list[str]:
        return [
            group
            for group, controls in self._group_controls.items()
            if controls["check"].isChecked()
        ]

    def _selected_uids(self) -> set[str]:
        return {
            uid
            for uid, enabled in self._cell_enabled_map().items()
            if enabled
        }

    def _group_color_map(self, groups: list[str] | None = None) -> dict[str, str]:
        ordered_groups = groups if groups is not None else list(self._group_controls.keys())
        colors: dict[str, str] = {}
        for group in ordered_groups:
            controls = self._group_controls.get(group)
            if not controls:
                continue
            button = controls.get("button")
            if button is None:
                continue
            colors[group] = str(button.property("color") or "#000000")
        return colors

    def _firing_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_bounds = tuple(sorted((float(self.firing_x_min.value()), float(self.firing_x_max.value()))))
        y_bounds = tuple(sorted((float(self.firing_y_min.value()), float(self.firing_y_max.value()))))
        return x_bounds, y_bounds

    def _access_resistance_max(self) -> float:
        return float(self.ra_max_spin.value())

    def _update_cell_summary(self) -> None:
        total = self.cell_table.rowCount()
        enabled = len(self._selected_uids())
        if total == 0:
            self.cell_summary_label.setText("No cells loaded yet.")
        else:
            self.cell_summary_label.setText(f"Enabled cells: {enabled} / {total}")

    def _on_cell_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_cell_table or item.column() != 0:
            return
        self._update_cell_summary()
        self._refresh_preview()

    def _filtered_data(self, prepared: dict[str, Any]) -> dict[str, Any]:
        base_cell_means = prepared["cell_means"]
        base_fi_cell = prepared["fi_cell"]
        selected_groups = set(self._selected_groups())
        selected_uids = self._selected_uids()

        if not selected_groups or not selected_uids or base_cell_means.empty:
            empty_cell_means = base_cell_means.iloc[0:0].copy()
            empty_fi_cell = base_fi_cell.iloc[0:0].copy() if isinstance(base_fi_cell, pd.DataFrame) else pd.DataFrame()
            return {
                "cell_means": empty_cell_means,
                "groups": [],
                "fi_stats": self._recompute_fi_stats(empty_fi_cell),
                "total_cells": pd.Series(dtype=int),
                "fi_cell": empty_fi_cell,
            }

        cell_mask = (
            base_cell_means["Group"].astype(str).isin(selected_groups)
            & base_cell_means["UID"].astype(str).isin(selected_uids)
        )
        filtered_cell_means = base_cell_means.loc[cell_mask].copy()
        remaining_groups = set(filtered_cell_means["Group"].astype(str).unique())
        visible_groups = [group for group in prepared["groups"] if group in remaining_groups]

        if isinstance(base_fi_cell, pd.DataFrame) and not base_fi_cell.empty:
            fi_mask = (
                base_fi_cell["Group"].astype(str).isin(visible_groups)
                & base_fi_cell["UID"].astype(str).isin(selected_uids)
            )
            filtered_fi_cell = base_fi_cell.loc[fi_mask].copy()
        else:
            filtered_fi_cell = pd.DataFrame(columns=["Group", "UID", "I_bin", "FiringRate"])

        total_cells = filtered_cell_means.groupby("Group")["UID"].nunique()
        total_cells = total_cells.reindex(visible_groups).fillna(0).astype(int)

        return {
            "cell_means": filtered_cell_means,
            "groups": visible_groups,
            "fi_stats": self._recompute_fi_stats(filtered_fi_cell),
            "total_cells": total_cells,
            "fi_cell": filtered_fi_cell,
        }

    def _recompute_fi_stats(self, fi_cell: pd.DataFrame) -> pd.DataFrame:
        if fi_cell is None or fi_cell.empty:
            return pd.DataFrame(columns=["Group", "I_bin", "mean", "sd", "n", "sem"])

        grouped = (
            fi_cell.groupby(["Group", "I_bin"], dropna=False)["FiringRate"]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        grouped["sem"] = grouped["sd"] / np.sqrt(grouped["n"].replace(0, np.nan))
        grouped["sem"] = grouped["sem"].fillna(0.0)
        return grouped
    def _create_plot(self, plot_key: str, filtered: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        group_colors = self._group_color_map(filtered["groups"])
        if plot_key == "FiringCurve":
            x_bounds, y_bounds = self._firing_bounds()
            fig, _ax, stats = grapher.plot_firing_curve(
                fi_stats=filtered["fi_stats"],
                groups=filtered["groups"],
                total_cells=filtered["total_cells"],
                per_cell=filtered["fi_cell"],
                group_colors=group_colors,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            return fig, stats

        palette = [group_colors.get(group, "#000000") for group in filtered["groups"]]
        fig, _ax, stats = grapher.plot_param(
            cell_means=filtered["cell_means"],
            groups=filtered["groups"],
            param=plot_key,
            colors=palette,
        )
        return fig, stats

    def _refresh_preview(self, *_args: object) -> None:
        if self.preview_combo.count() == 0:
            self._clear_preview("Select at least one plot category to preview.")
            return

        try:
            prepared = self._get_prepared_data(force_reload=False)
        except Exception as exc:
            self._clear_preview(str(exc))
            self.statusBar().showMessage("Unable to prepare graph data.")
            return

        filtered = self._filtered_data(prepared)
        plot_key = str(self.preview_combo.currentData() or "")

        if not filtered["groups"] or filtered["cell_means"].empty:
            self._clear_preview("No cells remain after the active access-resistance, group, and UID filters.")
            self.statusBar().showMessage("No data available for the current filters.")
            return

        if plot_key == "FiringCurve" and filtered["fi_cell"].empty:
            self._clear_preview("No firing-rate rows remain for the active filters.")
            self.statusBar().showMessage("No firing-rate data available for the current filters.")
            return

        try:
            figure, stats = self._create_plot(plot_key, filtered)
        except Exception as exc:
            self._clear_preview(f"Preview failed: {exc}")
            self.statusBar().showMessage("Preview failed.")
            return

        self._set_preview_figure(figure)
        self.stats_box.setPlainText(self._format_stats(plot_key, stats, filtered))
        self.statusBar().showMessage(f"Previewing {self._plot_label(plot_key)}.")

    def _set_preview_figure(self, figure: Any) -> None:
        self._clear_canvas_widgets()
        self._current_figure = figure
        self._current_canvas = FigureCanvasQTAgg(figure)
        self._current_toolbar = NavigationToolbar2QT(self._current_canvas, self)
        self.canvas_layout.addWidget(self._current_toolbar)
        self.canvas_layout.addWidget(self._current_canvas, 1)
        self._current_canvas.draw_idle()

    def _clear_canvas_widgets(self) -> None:
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if self._current_figure is not None:
            plt.close(self._current_figure)
        self._current_canvas = None
        self._current_toolbar = None
        self._current_figure = None

    def _clear_preview(self, message: str) -> None:
        self._clear_canvas_widgets()
        self._placeholder = QtWidgets.QLabel(message, self)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self.canvas_layout.addWidget(self._placeholder, 1)
        self.stats_box.setPlainText(message)

    def _format_stats(self, plot_key: str, stats: dict[str, Any], filtered: dict[str, Any]) -> str:
        def _fmt_number(value: Any) -> str:
            if value is None:
                return "NA"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return str(value)
            if np.isnan(numeric):
                return "NA"
            if abs(numeric) >= 1000 or (0 < abs(numeric) < 0.001):
                return f"{numeric:.3g}"
            if numeric.is_integer():
                return str(int(numeric))
            return f"{numeric:.4g}"

        lines = [self._plot_label(plot_key)]
        lines.append(f"Access resistance cutoff: <= {self._access_resistance_max():.1f} MOhm")
        lines.append(f"Included groups: {', '.join(filtered['groups'])}")
        lines.append(f"Included cells: {int(filtered['cell_means']['UID'].nunique())}")
        group_counts = filtered["total_cells"].to_dict() if not filtered["total_cells"].empty else {}
        if group_counts:
            counts_text = ", ".join(f"{group}={count}" for group, count in group_counts.items())
            lines.append(f"Cells by group: {counts_text}")

        if plot_key == "FiringCurve":
            lines.extend(self._firing_step_count_lines(filtered))
            anova = stats.get("anova") if isinstance(stats, dict) else None
            if isinstance(anova, dict) and anova.get("n", 0):
                lines.append(
                    "Two-way ANOVA: "
                    f"Group p={_fmt_number(anova.get('p_group'))}, "
                    f"Current p={_fmt_number(anova.get('p_current'))}, "
                    f"Interaction p={_fmt_number(anova.get('p_interaction'))}"
                )
            sidak = stats.get("sidak") if isinstance(stats, dict) else []
            significant = [entry for entry in sidak if str(entry.get("stars")) not in {"na", "ns"}]
            if significant:
                sig_text = ", ".join(
                    f"{_fmt_number(entry.get('I_bin'))} pA/pF ({entry.get('stars')})" for entry in significant
                )
                lines.append(f"Sidak significant steps: {sig_text}")
            elif isinstance(sidak, list):
                lines.append("Sidak significant steps: none")
            return "\n".join(lines)

        if not isinstance(stats, dict):
            return "\n".join(lines)

        lines.append(f"Test: {stats.get('test', 'NA')}")
        comparison = str(stats.get("comparison") or "").strip()
        if comparison:
            lines.append(f"Comparison: {comparison}")
        if stats.get("test") == "Unpaired t-test":
            lines.append(
                f"t={_fmt_number(stats.get('t'))}, df={_fmt_number(stats.get('df'))}, p={_fmt_number(stats.get('p'))}, stars={stats.get('stars', 'NA')}"
            )
        elif stats.get("test") == "One-way ANOVA":
            lines.append(
                f"F={_fmt_number(stats.get('F'))}, df1={_fmt_number(stats.get('df1'))}, df2={_fmt_number(stats.get('df2'))}, p={_fmt_number(stats.get('p'))}, stars={stats.get('stars', 'NA')}"
            )
        elif stats.get("test") == "Single group":
            lines.append("Single-group view; no between-group test was run.")

        return "\n".join(lines)

    def _firing_step_count_lines(self, filtered: dict[str, Any]) -> list[str]:
        fi_stats = filtered.get("fi_stats")
        groups = list(filtered.get("groups", []))
        if not isinstance(fi_stats, pd.DataFrame) or fi_stats.empty or not groups:
            return []

        x_bounds, _y_bounds = self._firing_bounds()
        x_lower, x_upper = x_bounds
        step_rows = fi_stats.loc[
            fi_stats["Group"].astype(str).isin(groups)
            & fi_stats["I_bin"].notna()
            & fi_stats["n"].notna()
        ].copy()
        if step_rows.empty:
            return []

        step_rows["I_bin"] = step_rows["I_bin"].astype(float)
        step_rows = step_rows.loc[step_rows["I_bin"].between(x_lower, x_upper)]
        if step_rows.empty:
            return []

        counts = (
            step_rows.pivot_table(
                index="I_bin",
                columns="Group",
                values="n",
                aggfunc="first",
                fill_value=0,
            )
            .reindex(columns=groups, fill_value=0)
            .sort_index()
        )

        lines = ["Step counts (cells per pA/pF):"]
        for step, row in counts.iterrows():
            step_value = float(step)
            step_text = str(int(step_value)) if step_value.is_integer() else f"{step_value:.4g}"
            parts = ", ".join(f"{group}={int(row[group])}" for group in groups)
            lines.append(f"{step_text} pA/pF: {parts}")
        return lines

    def _write_stats_text_file(self, save_dir: Path, stats_sections: list[str]) -> Path:
        separator = "\n\n" + ("=" * 72) + "\n\n"
        content = separator.join(section.rstrip() for section in stats_sections if section.strip())
        stats_path = save_dir / "PatchGrapher_stats.txt"
        stats_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return stats_path

    def _save_graphs(self) -> None:
        selected_plots = [key for key in PLOT_ORDER if self.plot_checks[key].isChecked()]
        if not selected_plots:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select at least one plot category to save.")
            return

        try:
            prepared = self._get_prepared_data(force_reload=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return

        filtered = self._filtered_data(prepared)
        if not filtered["groups"] or filtered["cell_means"].empty:
            QtWidgets.QMessageBox.information(self, "No data", "No cells remain after the active filters.")
            return

        save_dir = self._ensure_save_dir()
        saved_files: list[str] = []
        saved_stats_sections: list[str] = []
        group_colors = self._group_color_map(filtered["groups"])

        try:
            for plot_key in selected_plots:
                if plot_key == "FiringCurve":
                    if filtered["fi_cell"].empty:
                        continue
                    x_bounds, y_bounds = self._firing_bounds()
                    target = save_dir / "firing_curve.png"
                    fig, _ax, stats = grapher.plot_firing_curve(
                        fi_stats=filtered["fi_stats"],
                        groups=filtered["groups"],
                        total_cells=filtered["total_cells"],
                        savepath=str(target),
                        per_cell=filtered["fi_cell"],
                        group_colors=group_colors,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                    )
                    plt.close(fig)
                    saved_files.append(target.name)
                    saved_stats_sections.append(self._format_stats(plot_key, stats, filtered))
                    continue

                spec = grapher.PARAM_SPECS[plot_key]
                palette = [group_colors.get(group, "#000000") for group in filtered["groups"]]
                target = save_dir / spec.filename
                fig, _ax, stats = grapher.plot_param(
                    cell_means=filtered["cell_means"],
                    groups=filtered["groups"],
                    param=plot_key,
                    colors=palette,
                    savepath=str(target),
                )
                plt.close(fig)
                saved_files.append(target.name)
                saved_stats_sections.append(self._format_stats(plot_key, stats, filtered))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            self.statusBar().showMessage("Saving graphs failed.")
            return

        if not saved_files:
            QtWidgets.QMessageBox.information(self, "Nothing saved", "No graphs were saved for the current filters.")
            return

        stats_path = self._write_stats_text_file(save_dir, saved_stats_sections)

        QtWidgets.QMessageBox.information(
            self,
            "Graphs saved",
            f"Saved {len(saved_files)} graph(s) and stats text to:\n{save_dir}\n\nStats file: {stats_path.name}",
        )
        self.statusBar().showMessage(f"Saved {len(saved_files)} graph(s) and stats text to {save_dir}.")

    def _ensure_save_dir(self) -> Path:
        current = self.save_dir_edit.text().strip()
        if current:
            path = Path(current).expanduser()
        else:
            cc_path = Path(self.cc_path_edit.text().strip()).expanduser()
            if cc_path.is_file():
                path = cc_path.resolve().parent / "PatchGrapher_output"
            else:
                path = Path.cwd() / "PatchGrapher_output"
            self.save_dir_edit.setText(str(path))
        path.mkdir(parents=True, exist_ok=True)
        return path


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Patch Grapher")
    window = PatchGrapherWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
