from Image import *
from Histogram import *
from ImageDiffMethod import *
import numpy as np


class ImageComparison(GrayScaleTransform):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> 'Histogram':
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """

        return Histogram(self.data)


    def compare_to(self, other: GrayScaleTransform, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """

        if self.color_model != 4:
            first_photo = Histogram(self.to_gray_data()).values
        if other.color_model != 4:
            second_photo = Histogram(other.to_gray_data()).values
        else:
            first_photo = Histogram(self.data).values
            second_photo = Histogram(other.data).values

        mse = float()
        rmse = float()
        
        for element in range(len(first_photo)):
            mse += (1/len(first_photo)) * (first_photo[element] - second_photo[element]) ** 2
            result = np.sum(mse)

        if method == 1:
            rmse = result ** 0.5
            return rmse

        return result