"""
Task 1: Run an SPM-style GLM pipeline using Nilearn
Dataset: Nilearn language localizer demo dataset
Contrast: language (sentence reading) > string (consonant string reading)
"""

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving figures

import matplotlib.pyplot as plt
from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.glm.first_level import first_level_from_bids
from nilearn import plotting

# ---------- 1. Download the dataset ----------
data_dir = "./data"
data = fetch_language_localizer_demo_dataset(data_dir=data_dir)
data_dir = data.data_dir

# ---------- 2. Automatically extract FirstLevelModel ----------
# models, imgs, events, confounds = first_level_from_bids(
#     dataset_path=data_dir,
#     task_label="languagelocalizer",
#     smoothing_fwhm=8,
#     img_filters=[("desc", "preproc")],
#     n_jobs=1,
models, imgs, events, confounds = first_level_from_bids(
    dataset_path=data_dir,
    task_label="languagelocalizer",
    space_label="",
    smoothing_fwhm=8,
    img_filters=None,   # Set to None first, no filtering
    n_jobs=1,
)
# )

print(f"Number of runs extracted: {len(models)}")
print(f"Event table columns: {events[0][0].columns.tolist()}")
print(f"Confound columns: {confounds[0][0].columns.tolist()}")

# ---------- 3. Fit the model ----------
model = models[0]
model.fit(imgs[0], events=events[0], confounds=confounds[0])

# ---------- 4. Compute the contrast ----------
z_map = model.compute_contrast("language - string", output_type="z_score")

# ---------- 5. Visualize and save ----------
display = plotting.plot_stat_map(
    z_map,
    title="language > string (z-score)",
    display_mode="z",
    cut_coords=[0, 20, 40],
)
display.savefig("./outputs/task1_language_minus_string.png", dpi=150)
display.close()

print("Task 1 complete. Figure saved to outputs/")