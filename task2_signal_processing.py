"""
Task 2: Low-level signal processing of fMRI time series
Dataset: Nilearn language localizer demo dataset (same as Task 1)
Steps: extract ROI time series -> detrend -> bandpass filter -> z-score -> visualize
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from nilearn.maskers import NiftiSpheresMasker
from nilearn.datasets import fetch_language_localizer_demo_dataset
import os

# ---------- 1. Locate data ----------
data = fetch_language_localizer_demo_dataset(data_dir="./data")
data_dir = data.data_dir

func_file = os.path.join(
    data_dir, "derivatives", "sub-01", "func",
    "sub-01_task-languagelocalizer_desc-preproc_bold.nii.gz"
)
print("Functional image:", func_file)

# ---------- 2. Extract ROI time series ----------
seed_coords = [(0, -52, 18)]
masker = NiftiSpheresMasker(
    seeds=seed_coords,
    radius=8,
    t_r=1.5,           # TR read from BIDS sidecar (RepetitionTime)
    detrend=False,     # manual detrend below
    standardize=None,  # no automatic z-scoring, done manually below
)
raw_ts = masker.fit_transform(func_file).squeeze()
print("Raw time series length:", len(raw_ts))

# ---------- 3. Manual signal processing ----------
# 3a. Detrend: remove low-frequency scanner drift
detrended = detrend(raw_ts)

# 3b. Bandpass filter: keep 0.01-0.1 Hz task-related BOLD band
TR = 1.5
fs = 1.0 / TR
lowcut, highcut = 0.01, 0.1
nyq = fs / 2.0
b, a = butter(2, [lowcut / nyq, highcut / nyq], btype="band")
filtered = filtfilt(b, a, detrended)

# 3c. Standardize: z-score for amplitude comparability
standardized = (filtered - np.mean(filtered)) / np.std(filtered)

# ---------- 4. Visualize ----------
time = np.arange(len(raw_ts)) * TR

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(time, raw_ts, color="gray", linewidth=0.8)
axes[0].set_ylabel("Raw signal")
axes[0].set_title("Raw BOLD time series vs processed (detrend + bandpass + z-score)")

axes[1].plot(time, standardized, color="steelblue", linewidth=0.8)
axes[1].set_ylabel("Processed (z-score)")
axes[1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("./outputs/task2_signal_processing.png", dpi=150)
plt.close()

print("Task 2 done. Figure saved to outputs/task2_signal_processing.png")