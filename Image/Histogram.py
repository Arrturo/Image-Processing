import matplotlib.pyplot as plt
from GrayScaleTransform import *
import numpy as np


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """

    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:

        if values.ndim == 2:
            self.values, self.bin_edges = np.histogram(values, bins=256, range=(0, 256))

        if values.ndim == 3:
            self.values = np.zeros((256, 3))
            for layer_id in range(values.ndim):
                self.values[..., layer_id], self.bin_edges = np.histogram(values[..., layer_id], bins=256, range=(0, 256))

    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """

        if self.values.ndim == 2:
            colors = ("red", "green", "blue")
            f, rgb = plt.subplots(1, 3, figsize=(12, 4))
            for layer_id, color in enumerate(colors):
                rgb[layer_id].set_xlim([0, 256])
                rgb[layer_id].plot(self.bin_edges[:-1], self.values[..., layer_id], color=color)
            plt.tight_layout()

        if self.values.ndim == 1:
            plt.xlim([0, 256])
            plt.plot(self.bin_edges[:-1], self.values, color='gray')

        plt.show()

    def to_cumulated(self) -> 'Histogram':
        """
        metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        """

        if self.values.ndim == 1:
            self.values = np.cumsum(self.values)

        return self