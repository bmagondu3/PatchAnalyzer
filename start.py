# PatchAnalyzer/start.py
from __future__ import annotations
import sys
from PyQt5 import QtWidgets

from PatchAnalyzer.main_window import PatchAnalyzerGUI
from PatchAnalyzer.utils.log import setup_logger

# Call once so sub-modules inherit the handler hierarchy
setup_logger("PatchAnalyzer")

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = PatchAnalyzerGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
