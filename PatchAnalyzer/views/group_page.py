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
        self.meta_df = meta_df.copy()
        if "group_label" not in self.meta_df.columns:
            self.meta_df["group_label"] = ""

        # show just ONE representative row for every unique (x, y, z)
        self._rows: list[int] = (
            self.meta_df.groupby(["stage_x", "stage_y", "stage_z"])
            .head(1)
            .index
            .tolist()
        )
        self._groups: set[str] = set()          # all labels created so far

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
        """Fill the table then enable click-to-sort on any column."""
        # 1) turn sorting OFF while we insert rows (avoids flicker / bugs)
        self.table.setSortingEnabled(False)

        self.table.setColumnCount(len(self._COLS))
        self.table.setHorizontalHeaderLabels(
            [c.replace("_", " ").title() for c in self._COLS]
        )
        self.table.setRowCount(len(self._rows))

        for row_idx, df_idx in enumerate(self._rows):
            row = self.meta_df.loc[df_idx]
            for col_idx, col in enumerate(self._COLS):
                if col == "src_dir":
                    display_val = Path(row["src_dir"]).name  # folder only
                else:
                    display_val = str(row[col])

                item = QtWidgets.QTableWidgetItem(display_val)

                # lock index & label cells from manual edits
                if col in ("index", "group_label"):
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)

                if col == "group_label":
                    self._style_group_item(item, display_val)

                self.table.setItem(row_idx, col_idx, item)

        # 2) NOW enable sorting and make header clickable/with arrow
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.setSortingEnabled(True)

    # ----------------------------------------------------------------- label
    def _assign_label(self) -> None:
        """
        Ask the user for a group label and apply it to all selected rows.

        • If at least one label already exists, we present a combo box that
          lists the current labels *and* allows free-text entry (editable=True).
        • If no labels exist yet, we fall back to a plain text prompt.
        """
        sel_rows = self.table.selectionModel().selectedRows()
        if not sel_rows:
            QtWidgets.QMessageBox.information(
                self, "Nothing selected", "Select one or more rows first."
            )
            return

        label: str = ""

        if self._groups:  # show existing labels + allow new ones
            label, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Group Label",
                "Select an existing label or type a new one:",
                sorted(self._groups),
                0,
                True,  # editable
            )
            if not ok:
                return
            label = label.strip()
        else:  # first label ever → simple text box
            label, ok = QtWidgets.QInputDialog.getText(
                self,
                "Group Label",
                "Enter a label:",
                QtWidgets.QLineEdit.Normal,
                "",
            )
            if not ok:
                return
            label = label.strip()

        if not label:
            return  # user entered nothing

        # record new label
        self._groups.add(label)
        group_col = self._COLS.index("group_label")

        for model_index in sel_rows:
            tbl_row = model_index.row()
            df_idx = self._rows[tbl_row]

            # update DataFrame
            self.meta_df.at[df_idx, "group_label"] = label

            # update table cell
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
