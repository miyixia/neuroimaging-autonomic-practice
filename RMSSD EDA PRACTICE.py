"""
Autonomic signal processing practice (simulated data)
1. RMSSD: compute a parasympathetic index from RR intervals
2. EDA: simulate skin conductance data, detrend, and decompose into tonic/phasic components

Dependencies: numpy, scipy, matplotlib, neurokit2
Install: pip install numpy scipy matplotlib neurokit2
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend
import neurokit2 as nk

# ============================================================
# Part 1: RMSSD calculation (simulated RR intervals)
# ============================================================
# Simulate an RR interval series (unit: ms); resting values are typically 800-1000 ms
# Here we simulate 60 beats with natural variability
np.random.seed(42)
rr_intervals = 850 + np.random.randn(60) * 30  # mean 850 ms, SD 30 ms

# RMSSD formula: sqrt(mean(diff(RR)^2))
# RMSSD reflects short-term heart rate variability and is mainly linked to parasympathetic modulation
diff_rr = np.diff(rr_intervals)
rmssd = np.sqrt(np.mean(diff_rr ** 2))

print("[RMSSD practice]")
print(f"  Number of simulated RR intervals: {len(rr_intervals)}")
print(f"  RMSSD = {rmssd:.2f} ms")
print(f"  Interpretation: higher RMSSD generally indicates stronger parasympathetic (vagal) modulation.\n")

###############
"""
RMSSD practice on raw ECG data (NSRDB, first 10 minutes)
Pipeline: raw ECG -> signal cleaning -> R-peak detection with artifact correction -> RR intervals -> RMSSD
"""

import wfdb
import numpy as np
import neurokit2 as nk

# ============================================================
# 1. Load raw ECG signal (first 10 minutes)
# ============================================================
RECORD = "16265"
FS = 128  # NSRDB sampling rate (Hz)
DURATION_S = 10 * 60
MAX_SAMPLE = FS * DURATION_S

record = wfdb.rdrecord(RECORD, pn_dir="nsrdb", sampto=MAX_SAMPLE)
ecg_raw = record.p_signal[:, 0]  # use ECG1 channel
print(f"[Data] Record: {RECORD}, samples: {len(ecg_raw)}, duration: {len(ecg_raw)/FS/60:.1f} min")

# ============================================================
# 2. Signal cleaning (first step in a realistic pipeline)
# ============================================================
# ecg_clean uses the NeuroKit method by default: 0.5 Hz high-pass filter + powerline notch,
# removing baseline drift and high-frequency noise
ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=FS)
print(f"[Cleaning] ecg_clean completed (0.5 Hz high-pass + notch)")

# ============================================================
# 3. R-peak detection (with artifact correction)
# ============================================================
# correct_artifacts=True invokes the Lipponen & Tarvainen (2019) algorithm to
# identify and correct ectopic beats, missed beats, and false detections
signals, info = nk.ecg_peaks(
    ecg_cleaned,
    sampling_rate=FS,
    method="neurokit",
    correct_artifacts=True,
)
r_peaks = info["ECG_R_Peaks"]
print(f"[Detection] Number of R peaks (after correction): {len(r_peaks)}")

# ============================================================
# 4. Convert R-peak positions to RR intervals (ms)
# ============================================================
rr_intervals = np.diff(r_peaks) / FS * 1000  # ms

# ============================================================
# 5. Quality control checks
# ============================================================
rr_min, rr_max = 500, 1500
n_outlier = np.sum((rr_intervals < rr_min) | (rr_intervals > rr_max))
print(f"[QC] Number of RR intervals: {len(rr_intervals)}")
print(f"[QC] Mean RR: {rr_intervals.mean():.1f} ms")
print(f"[QC] Min/Max RR: {rr_intervals.min():.1f} / {rr_intervals.max():.1f} ms")
print(f"[QC] Implausible RR intervals: {n_outlier}")

# ============================================================
# 6. Compute RMSSD
# ============================================================
def compute_rmssd(rr):
    return np.sqrt(np.mean(np.diff(rr) ** 2))

rmssd = compute_rmssd(rr_intervals)
print(f"\n[RMSSD] Corrected series: {rmssd:.2f} ms")


# ============================================================
# Part 2: EDA detrending and tonic/phasic decomposition (simulated signal)
# ============================================================
# Simulate an EDA signal: slow drift (tonic) + fast SCR fluctuations (phasic) + noise
fs = 100  # sampling rate in Hz (commonly 100-1000)
duration = 60  # seconds
t = np.arange(0, duration, 1/fs)

# Construct tonic (slow drift) and phasic (SCR pulses) components
tonic_sim = 2.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t)  # very slow drift
phasic_sim = np.zeros_like(t)
for onset in [10, 25, 40]:  # simulate SCR events at 10 s, 25 s, 40 s
    idx = (t >= onset) & (t < onset + 5)
    phasic_sim[idx] = 0.5 * np.exp(-(t[idx] - onset) / 1.5)

# Combine into a raw EDA signal (with a small amount of noise)
eda_raw = tonic_sim + phasic_sim + 0.02 * np.random.randn(len(t))

# Method 1: manual detrending (remove linear trend, approximating the tonic component)
eda_detrended = detrend(eda_raw)

# Method 2: use NeuroKit2 for tonic/phasic decomposition (recommended, more standard)
# eda_phasic uses a high-pass filter (~0.05 Hz) to separate phasic from tonic
signals = nk.eda_phasic(eda_raw, sampling_rate=fs)
eda_tonic = signals["EDA_Tonic"]
eda_phasic = signals["EDA_Phasic"]

print("[EDA processing practice]")
print(f"  Raw EDA length: {len(eda_raw)} samples ({duration} s)")
print(f"  Mean tonic (SCL): {eda_tonic.mean():.3f}")
print(f"  SD of phasic (SCR): {eda_phasic.std():.3f}")
print(f"  Note: the phasic component reflects fast skin conductance responses (SCRs), an index of sympathetic arousal.\n")

# ============================================================
# Part 3: Visualization
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1. Raw EDA
axes[0].plot(t, eda_raw, color="gray", linewidth=0.8)
axes[0].set_ylabel("Raw EDA (µS)")
axes[0].set_title("EDA Processing Practice: Raw vs Tonic/Phasic")

# 2. Detrended EDA (manual)
axes[1].plot(t, eda_detrended, color="steelblue", linewidth=0.8)
axes[1].set_ylabel("Detrended")
axes[1].set_title("Manual detrend (linear trend removed)")

# 3. NeuroKit2 tonic/phasic decomposition
axes[2].plot(t, eda_tonic, color="green", label="Tonic (SCL)", linewidth=1)
axes[2].plot(t, eda_phasic, color="orange", label="Phasic (SCR)", linewidth=0.8)
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("NeuroKit2 eda_phasic: Tonic vs Phasic decomposition")
axes[2].legend()

plt.tight_layout()
plt.savefig("./outputs/eda_rmssd_practice.png", dpi=150)
plt.close()

print("[Output] Comparison figure saved to ./outputs/eda_rmssd_practice.png")
print("[Note] RMSSD computed with numpy diff/sqrt; EDA decomposition via NeuroKit2 high-pass filter (~0.05 Hz).")

###_____________Processing of raw EDA data
# 1. Inspect the file contents
from scipy.io import loadmat
# Change the path to the actual file location
data = loadmat("D:/miyix/Downloads/Electrodermal activity - E4/EDA/SADNESS/39_SadnessList_.mat", simplify_cells=True)
print(data.keys())
for key in ['fileinfo', 'data', 'analysis']:
    print(f"\n===== {key} =====")
    print(type(data[key]))
    if isinstance(data[key], dict):
        print("Sub-keys:", list(data[key].keys()))
    else:
        print("Shape:", getattr(data[key], 'shape', 'no shape attribute'))
        print("First rows/values:", data[key] if not hasattr(data[key], 'shape') else data[key][:3])

# Check shape and value ranges
import numpy as np

print("conductance shape:", data['data']['conductance'].shape)
print("conductance first 5 values:", np.ravel(data['data']['conductance'])[:5])
print("time shape:", data['data']['time'].shape)
print("time first 5 values:", np.ravel(data['data']['time'])[:5])
print("time last 5 values:", np.ravel(data['data']['time'])[-5:])

# Check overall range
cond = np.ravel(data['data']['conductance'])
print("Min:", cond.min())
print("Max:", cond.max())
print("Mean:", cond.mean())
print("SD:", cond.std())

### Baseline correction

import neurokit2 as nk

cond = np.ravel(data['data']['conductance'])
cond_corrected = cond - cond.min()  # baseline correction, shifting the minimum close to 0

# Confirm the corrected range
print("Corrected min:", cond_corrected.min())
print("Corrected max:", cond_corrected.max())
print("Corrected mean:", cond_corrected.mean())

##### Run eda_process
signals, info = nk.eda_process(cond_corrected, sampling_rate=10)

print("Number of detected SCRs:", len(info.get("SCR_Peaks", [])))
scr_amp = info.get("SCR_Amplitude", [])
print("Mean SCR amplitude:", np.mean(scr_amp) if len(scr_amp) > 0 else "none")
print("Mean tonic:", signals["EDA_Tonic"].mean())
print("SD of phasic:", signals["EDA_Phasic"].std())

#### Continuous signal
import numpy as np
import neurokit2 as nk

# cond_corrected is the baseline-corrected EDA signal from above, fs = 10
signals, info = nk.eda_process(cond_corrected, sampling_rate=10)

# Continuous phasic time series (same length as the original signal)
phasic_continuous = signals["EDA_Phasic"].values
tonic_continuous = signals["EDA_Tonic"].values

print("Phasic length:", len(phasic_continuous), "should match original signal:", len(cond_corrected))
print("Phasic range:", phasic_continuous.min(), phasic_continuous.max())
print("Phasic mean:", phasic_continuous.mean())

author_amp = np.ravel(data['analysis']['amp'])
print("Number of author SCRs:", len(author_amp))
print("Author SCR amplitude range:", author_amp.min(), author_amp.max())
print("Author mean SCR amplitude:", author_amp.mean())

#####

# Length of the author's tonicData
tonic_author = np.ravel(data['analysis']['tonicData'])
print("Author tonicData length:", len(tonic_author))
print("Your conductance length:", len(cond_corrected))

# Range of the author's peakTime
peak_time = np.ravel(data['analysis']['peakTime'])
print("Author peakTime range:", peak_time.min(), peak_time.max())
print("Number of author peakTimes:", len(peak_time))

# Method used by the authors
print("Author method:", data['analysis']['method'])