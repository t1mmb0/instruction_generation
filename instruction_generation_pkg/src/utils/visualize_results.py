import matplotlib.pyplot as plt
import numpy as np
import os 
from configs import paths
# -----------------------
# Visualizer Class
# -----------------------


class Visualizer:
    def __init__(self, history_train: list = None, history_val: list = None, file_path = paths.CONFIG["paths"]["plots"], smoothing_window = -1):
        self.history = {
            "train": history_train,
            "val": history_val,
        }
        self.file_path = file_path
        self.smoothing_window = smoothing_window

    def plot_all(self,):
        #Platzhalter: Später Methode, um alle Plts zusammen zu plotten und zu speichern.
        pass

    def plot_loss(self, show: bool = True):
        for name, values in self.history.items():
            if values is None:
                continue
            if self.smoothing_window > 0:
                values = self._smooth_plt(values=values)
            plt.plot(values, label=f"{name}-loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(os.path.join(self.file_path, f"loss_curve.png"), dpi=300, bbox_inches="tight")
        print("plot saved successfully!")
        if show:
            plt.show()
    
    def _smooth_plt(self, values,):
        if values is None or len(values) < self.smoothing_window:
            return values
        return np.convolve(values, np.ones(self.smoothing_window)/self.smoothing_window, mode="valid")
