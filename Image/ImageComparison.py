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

    def compare_to(self, other: GrayScaleTransform, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """

        first_photo = self.to_gray().histogram().values
        second_photo = other.to_gray().histogram().values

        mse = float()
        rmse = float()
        
        for element in range(len(first_photo)):
            mse += (1/len(first_photo)) * (first_photo[element] - second_photo[element]) ** 2
            result = np.sum(mse)

        if method == 1:
            rmse = result ** 0.5
            return rmse

        return result