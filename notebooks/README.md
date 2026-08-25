# notebooks/

Scratch space only -- exploration, plotting, Grad-CAM, demo inference.

Nothing here is imported by `model.py` / `train.py` and nothing here is the
source of truth. If a notebook cell becomes something you rely on across runs,
move it into `model.py` and drive it from a config.

Typical use from a Colab cell after a run:

```python
import pandas as pd, matplotlib.pyplot as plt
h = pd.read_csv("experiments/test13_face/history.csv")
h[["accuracy", "val_accuracy"]].plot(); plt.show()
h[["auc", "val_auc"]].plot(); plt.show()
```
