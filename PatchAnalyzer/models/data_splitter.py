# #PatchAnalyzer/models/data_splitter.py
# import os
# from pathlib import Path

# # === CONFIG ===
# directory = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\2025_06_28-14_16\HoldingProtocol"  # Change to your directory
# num_splits =  3 # Change to how many equal splits you want
# has_header = False  # Set to True if the CSV has a header line to repeat in each split
# # ==============

# def split_file(file_path, splits, has_header=False):
#     stem = file_path.stem
#     suffix = file_path.suffix

#     # Count total lines
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     header_line = None
#     if has_header:
#         header_line = lines[0]
#         lines = lines[1:]

#     total = len(lines)
#     chunk_size = total // splits
#     remainder = total % splits

#     start_idx = 0
#     for i in range(1, splits + 1):
#         end_idx = start_idx + chunk_size + (1 if i <= remainder else 0)
#         out_file = file_path.with_name(f"{stem}_{i}{suffix}")
#         with open(out_file, "w", encoding="utf-8") as out:
#             if header_line:
#                 out.write(header_line)
#             out.writelines(lines[start_idx:end_idx])
#         start_idx = end_idx

#     print(f"Split {file_path.name} into {splits} parts.")

# def main():
#     dir_path = Path(directory)

#     for file_path in dir_path.glob("HoldingProtocol_*_k.csv"):
#         # Skip if already processed (has an extra _number before .csv)
#         if "_k_" in file_path.stem:
#             continue
#         split_file(file_path, num_splits, has_header)

# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np

# file = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\HoldingProtocol\splits\HoldingProtocol_10_k_3.csv"  # change this to your file path
# file = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\CurrentProtocol\CurrentProtocol_1_#849cac_60.0.csv"
file = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\VoltageProtocol\VoltageProtocol_1_k.csv"

data = np.loadtxt(file)  # loads as numpy array
x, y, z = data[:,0], data[:,1], data[:,2]

# Create 2x1 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# First subplot
ax1.plot(x, y)
ax1.set_xlabel("Column 1")
ax1.set_ylabel("Column 2")
ax1.grid(True)

# Second subplot
ax2.plot(x, z)
ax2.set_xlabel("Column 1")
ax2.set_ylabel("Column 3")
ax2.grid(True)

plt.tight_layout()
plt.show()