import matplotlib.pyplot as plt
from GrayScaleTransform import *
import numpy as np


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, photo: BaseImage) -> None:
        self.color_model = photo.color_model
        if self.color_model == 4:
            self.values, self.bin_edges = np.histogram(photo.data, bins=256, range=(0, 256))

        if self.color_model == 0:
            self.values = np.zeros((256, 3))
            for layer_id in range(3):
                self.values[..., layer_id], self.bin_edges = np.histogram(photo.data[..., layer_id], bins=256, range=(0, 256))

    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """

        if self.color_model == 0:
            colors = ("red", "green", "blue")
            f, rgb = plt.subplots(1, 3, figsize=(12, 4))
            for layer_id, color in enumerate(colors):
                rgb[layer_id].set_xlim([0, 256])
                rgb[layer_id].plot(self.bin_edges[:-1], self.values[..., layer_id], color=color)
            plt.tight_layout()

        if self.color_model == 4:
            plt.xlim([0, 256])
            plt.plot(self.bin_edges[:-1], self.values, color='gray')

        plt.show()
