import numpy as np
import matplotlib.pyplot as plt

filenames = [
    "vhu3LxxG-Tg_0",
    "Q3roIYzQSGo_1",
    "3sNqcTzdWMU_1",
    "zWpoSvLmrW4_0",
    "pVYn7eRNTCg_1",
    "iWE-vOq4Cj8_0",
    "lNVVNxtolyE_1",
    "h0HjktaeEpA_0",
]
metric_1 = [
    43.268587,
    6.430380,
    0.000000,
    0.653431,
    7.539118,
    4.440877,
    0.000000,
    12.656467,
]
metric_2 = [
    56.932351,
    9.822785,
    19.024857,
    13.370194,
    19.061166,
    7.490637,
    5.578343,
    27.955494,
]

fig1, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(filenames))
width = 0.4

rects1 = ax1.bar(
    x - width / 2,
    metric_1,
    width,
    label="Audio-Only Diarisation",
    color="b",
    edgecolor="black",
)
rects2 = ax1.bar(
    x + width / 2,
    metric_2,
    width,
    label="Multimodal Diarisation",
    color="r",
    edgecolor="black",
)

ax1.set_ylabel("Confusion (%)")
ax1.set_xlabel("Clip ID")
ax1.set_xticks(x)
ax1.set_xticklabels(filenames, rotation=45, ha="right")
ax1.legend()

# Adding data labels on top of bars
# ax1.bar_label(rects1, padding=3, fontsize=9)
# ax1.bar_label(rects2, padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("barchart_confusion.png")
