from Image import *
from Histogram import *
from ImageDiffMethod import *
import numpy as np


class ImageComparison(BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> 'Histogram':
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """

        return Histogram(self)

    def compare_to(self, other: BaseImage, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """

        first_photo = self.histogram().values
        second_photo = Histogram(other).values
        mse = float()

        if method == 0:
            for element in range(len(first_photo)):
                mse = (1 / 256) * np.sum(first_photo[element] - second_photo[element]) ** 2
            return mse

        if method == 1:
            for element in range(len(first_photo)):
                rmse = np.sqrt((1 / 256) * np.sum(first_photo[element] - second_photo[element]) ** 2)
            return rmse