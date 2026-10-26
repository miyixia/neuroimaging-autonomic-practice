"""
任务 1：用 Nilearn 跑通 SPM 风格的 GLM 流程
数据集：Nilearn 官方语言定位器演示数据集
对比：language（读句子）> string（读辅音串）
"""

import matplotlib
matplotlib.use("Agg")  # 无界面保存图片

import matplotlib.pyplot as plt
from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.glm.first_level import first_level_from_bids
from nilearn import plotting

# ---------- 1. 下载数据集 ----------
data_dir = "./data"
data = fetch_language_localizer_demo_dataset(data_dir=data_dir)
data_dir = data.data_dir

# ---------- 2. 自动提取 FirstLevelModel ----------
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
    img_filters=None,   # 先设为 None，不做过滤
    n_jobs=1,
)
# )

print(f"共提取到 {len(models)} 个 run")
print(f"事件表列名：{events[0][0].columns.tolist()}")
print(f"混淆回归量列名：{confounds[0][0].columns.tolist()}")

# ---------- 3. 拟合模型 ----------
model = models[0]
model.fit(imgs[0], events=events[0], confounds=confounds[0])

# ---------- 4. 计算 contrast ----------
z_map = model.compute_contrast("language - string", output_type="z_score")

# ---------- 5. 可视化并保存 ----------
display = plotting.plot_stat_map(
    z_map,
    title="language > string (z-score)",
    display_mode="z",
    cut_coords=[0, 20, 40],
)
display.savefig("./outputs/task1_language_minus_string.png", dpi=150)
display.close()

print("任务 1 完成，结果图已保存到 outputs/")