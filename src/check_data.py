import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

root = Path("data/cfp_raw/translated_data")
names = sorted(p.stem for p in (root / "images").glob("*.png"))

bm = np.array(Image.open(root / "binary_mask" / f"{names[0]}.png"))
print("binary_mask shape", bm.shape, "dtype", bm.dtype)

id_counts, colors = {}, {}
for n in names[:100]:
    m = np.array(Image.open(root / "binary_mask" / f"{n}.png"))
    cm = np.array(Image.open(root / "color_mask" / f"{n}.png"))
    for v, c in zip(*np.unique(m, return_counts=True)):
        id_counts[int(v)] = id_counts.get(int(v), 0) + int(c)
        if int(v) not in colors:
            colors[int(v)] = tuple(int(x) for x in cm[m == v][0])
total = sum(id_counts.values())
print("class id | pixel share | RGB in color_mask")
for k in sorted(id_counts):
    print(k, "|", round(100 * id_counts[k] / total, 2), "% |", colors[k])

random.seed(0)
sample = random.sample(names, 4)
fig, axes = plt.subplots(len(sample), 3, figsize=(13, 4 * len(sample)))
for i, n in enumerate(sample):
    img = np.array(Image.open(root / "images" / f"{n}.png"))
    m = np.array(Image.open(root / "binary_mask" / f"{n}.png"))
    cm = np.array(Image.open(root / "color_mask" / f"{n}.png"))
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(cm)
    axes[i, 2].imshow(img); axes[i, 2].imshow(cm, alpha=0.5)
    for a, t in zip(axes[i], ["image", "color_mask", "overlay"]):
        a.set_title(t); a.axis("off")
plt.tight_layout()
plt.savefig("results/data_check.png", dpi=80)
plt.show()
