from typing import Optional
from BaseImage import BaseImage
import numpy as np


class ImageFiltration:

    @staticmethod
    def conv_2d(image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """

        if prefix is None:
            prefix = 1.0

        if image.color_model == 4:
            output_image = np.zeros((image.data.shape[0], image.data.shape[1]))

            for i in range(image.data.shape[0]):
                for j in range(image.data.shape[1]):

                    for m in range(kernel.shape[0]):
                        for n in range(kernel.shape[1]):
                            output_image[i][j] += kernel[m][n] * image.data[i - m][j - n]

        image.data = output_image * prefix
        return image