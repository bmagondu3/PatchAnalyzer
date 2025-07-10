# PatchAnalyzer/views/group_page.py
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets


class GroupPage(QtWidgets.QWidget):
    """
    Page 2  –  tabulate the unique cells and let the user assign group labels,
    optionally saving the result as a CSV.

    Signals
    -------
    back_requested → user pressed  ← Back
    done(DataFrame) → user pressed Continue ▶  (DataFrame has “group_label” col)
    """
    back_requested = QtCore.pyqtSignal()
    done = QtCore.pyqtSignal(pd.DataFrame)

    # --------------------------------------------------------------------- init
    def __init__(self, meta_df: pd.DataFrame, parent=None):
        super().__init__(parent)

        # make a working copy; ensure the “group_label” column exists
        self.meta_df = meta_df.copy()
        if "group_label" not in self.meta_df.columns:
            self.meta_df["group_label"] = ""

        # ── coordinate → list[indices] mapping ───────────────────────────
        self._groups_by_coord: dict[tuple[float, float, float], list[int]] = (
            self.meta_df.groupby(["stage_x", "stage_y", "stage_z"]).indices
        )

        # one representative DataFrame index per cell (first in each list)
        self._rows: list[int] = [idxs[0] for idxs in self._groups_by_coord.values()]

        # existing labels (to pre-populate the combo later)
        self._groups: set[str] = set(
            self.meta_df.loc[self.meta_df["group_label"].ne(""), "group_label"].unique()
        )

        # build UI & fill table
        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------  UI
    def _build_ui(self) -> None:
        base = QtWidgets.QVBoxLayout(self)
        base.setContentsMargins(8, 8, 8, 8)

        # ── table ────────────────────────────────────────────────────────
        self.table = QtWidgets.QTableWidget(
            selectionBehavior=QtWidgets.QAbstractItemView.SelectRows,
            selectionMode=QtWidgets.QAbstractItemView.ExtendedSelection,
        )
        self.table.horizontalHeader().setStretchLastSection(True)


        self.table.setSortingEnabled(True)

        base.addWidget(self.table, stretch=1)

        # ── bottom buttons ───────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        base.addLayout(btn_row)

        self.btn_assign = QtWidgets.QPushButton("Assign Label…")
        self.btn_assign.clicked.connect(self._assign_label)
        btn_row.addWidget(self.btn_assign)

        self.btn_save = QtWidgets.QPushButton("Save CSV…")
        self.btn_save.clicked.connect(self._save_csv)
        btn_row.addWidget(self.btn_save)

        btn_row.addStretch(1)

        self.btn_back = QtWidgets.QPushButton("← Back")
        self.btn_back.clicked.connect(self.back_requested)
        btn_row.addWidget(self.btn_back)

        self.btn_continue = QtWidgets.QPushButton("Continue ▶")
        self.btn_continue.clicked.connect(self._on_continue)
        btn_row.addWidget(self.btn_continue)

    # ---------------------------------------------------- table population
    _COLS = ["index", "stage_x", "stage_y", "stage_z", "src_dir", "group_label"]

    def _populate_table(self) -> None:
        """
        Show one row per unique (x,y,z) cell.

        • “index” column now lists the **values from the CSV “index” field**
          (e.g. 0, 1, 2 …) – not the DataFrame row numbers.
        """
        self.table.setSortingEnabled(False)

        self.table.setColumnCount(len(self._COLS))
        self.table.setHorizontalHeaderLabels(
            [c.replace("_", " ").title() for c in self._COLS]
        )
        self.table.setRowCount(len(self._rows))

        for tbl_row, rep_df_idx in enumerate(self._rows):
            rep = self.meta_df.loc[rep_df_idx]

            # all DataFrame row-ids for this coordinate triple
            df_row_list = self._groups_by_coord[
                (rep.stage_x, rep.stage_y, rep.stage_z)
            ]
            # corresponding *CSV index* values
            cell_idx_vals = [
                str(self.meta_df.at[i, "index"]) for i in df_row_list
            ]

            col_vals = {
                "index": ", ".join(cell_idx_vals),
                "stage_x": f"{rep.stage_x:.2f}",
                "stage_y": f"{rep.stage_y:.2f}",
                "stage_z": f"{rep.stage_z:.2f}",
                "src_dir": Path(rep["src_dir"]).name,
                "group_label": rep["group_label"],
            }

            for col_idx, col in enumerate(self._COLS):
                item = QtWidgets.QTableWidgetItem(col_vals[col])
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)

                if col == "group_label" and col_vals[col]:
                    self._style_group_item(item, col_vals[col])

                self.table.setItem(tbl_row, col_idx, item)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
   # ----------------------------------------------------------------- label
    def _assign_label(self) -> None:
        """
        Prompt for a group label and apply it to every DataFrame row that shares
        the coordinates of each selected table row.
        """
        sel_rows = self.table.selectionModel().selectedRows()
        if not sel_rows:
            QtWidgets.QMessageBox.information(
                self, "Nothing selected", "Select one or more rows first."
            )
            return

        # ask for label (reuse existing labels if any)
        if self._groups:
            label, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Group Label",
                "Select an existing label or type a new one:",
                sorted(self._groups),
                0,
                True,
            )
        else:
            label, ok = QtWidgets.QInputDialog.getText(
                self,
                "Group Label",
                "Enter a label:",
                QtWidgets.QLineEdit.Normal,
                "",
            )
        label = label.strip()
        if not ok or not label:
            return

        self._groups.add(label)
        group_col = self._COLS.index("group_label")

        for model_index in sel_rows:
            tbl_row = model_index.row()
            rep_df_idx = self._rows[tbl_row]

            # all rows with the same (x,y,z)
            coord = tuple(self.meta_df.loc[rep_df_idx][["stage_x", "stage_y", "stage_z"]])
            mask = (
                (self.meta_df["stage_x"] == coord[0]) &
                (self.meta_df["stage_y"] == coord[1]) &
                (self.meta_df["stage_z"] == coord[2])
            )
            self.meta_df.loc[mask, "group_label"] = label

            # update table cell for representative row
            item = self.table.item(tbl_row, group_col)
            item.setText(label)
            self._style_group_item(item, label)

    # ----------------------------------------------------------------- save
    def _save_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save grouped data as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            self.meta_df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved",
                                              f"Groups saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error",
                                           f"Failed to save CSV:\n{exc}")

    # ------------------------------------------------------------ continue
    def _on_continue(self) -> None:
        self.done.emit(self.meta_df)

    # ---------------------------------------------------------- aesthetics
    def _style_group_item(self, item: QtWidgets.QTableWidgetItem, label: str) -> None:
        """Draw a light-coloured “pill” for the group label cell."""
        if not label:
            return
        hue = int(hashlib.md5(label.encode()).hexdigest(), 16) % 360
        color = QtGui.QColor.fromHsl(hue, 160, 200)
        item.setBackground(color)
        item.setForeground(QtGui.QColor("black"))
        item.setTextAlignment(QtCore.Qt.AlignCenter)
