from typing import Optional
from BaseImage import BaseImage
import numpy as np


class ImageFiltration:
    @staticmethod
    def __conv_channel(image: np.ndarray, kernel: np.ndarray, prefix: Optional[float] = None) -> np.ndarray:
        output_image = np.zeros((image.data.shape[0], image.data.shape[1]))

        if prefix is None:
            prefix = 1

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        output_image[i][j] += kernel[m][n] * image[i - m][j - n]

        output_image *= prefix
        output_image[output_image < 0] = 0
        output_image[output_image > 255] = 255
        return output_image.astype('uint8')


    @staticmethod
    def conv_2d(image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """

        if image.color_model == 4:
            image.data = ImageFiltration.__conv_channel(image.data, kernel, prefix)

        if image.color_model == 0:
            R = ImageFiltration.__conv_channel(image.data[:, :, 0], kernel, prefix)
            G = ImageFiltration.__conv_channel(image.data[:, :, 1], kernel, prefix)
            B = ImageFiltration.__conv_channel(image.data[:, :, 2], kernel, prefix)

            image.data = np.dstack((R, G, B))
        return image

