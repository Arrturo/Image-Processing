import numpy as np
from Histogram import *
import cv2


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """

    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)

    @staticmethod
    def __align_channel(channel: np.ndarray, tail_elimination: bool = False) -> np.ndarray:
        """
        metoda wyrównująca histogram danego kanału
        """
        channel = channel.astype(np.float64)
        max_value = np.max(channel)
        min_value = np.min(channel)

        if tail_elimination == True:
            max_value = np.quantile(channel, 0.95)
            min_value = np.quantile(channel, 0.05)

        channel = ((channel - min_value) / (max_value - min_value)) * 255

        channel[channel > 255] = 255
        channel[channel < 0] = 0

        return channel.astype('uint8')


    def align_image(self, tail_elimination: bool = False) -> 'BaseImage':
        """
        metoda wyrównująca histogram obrazu
        """
        if self.color_model == 0:
            R = self.__align_channel(self.data[..., 0], tail_elimination)
            G = self.__align_channel(self.data[..., 1], tail_elimination)
            B = self.__align_channel(self.data[..., 2], tail_elimination)

            self.data = np.dstack((R, G, B))
                # self.data[:, :, i] = self.align_channel(img_data, tail_elimination)

        if self.color_model == 4:
            self.data = self.__align_channel(self.data, tail_elimination)
        return self


    def clache(self, clipLimit, tileGridSize) -> BaseImage:
        """
        Metoda CLAHE polega adaptywnym na ograniczaniu wysokich wartości na histogramie.
        Po dokonaniu operacji wyrównana zostaje redystrybucja krańcowych wartości pikseli po całym obszarze obrazu.
        Metoda CLAHE znacząco obniża również liczebnośc pikseli o wartościach granicznych.
        """

        if self.color_model == 4:
            # clipLimit - wartość progu do limitowania kontrastu (wielkość odstawania barwy do zredukowania) tileGridSize
            # - wielkość pojedynczych fragmentów, w ktorych wyrownywany jest histogram
            # (wielkosc sasiedztwa ekstremów na histogramie)

            clahe = cv2.createCLAHE(clipLimit, tileGridSize) #clipLimit=2.0, tileGridSize=(4, 4)
            self.data = clahe.apply(self.data)

        if self.color_model == 0:
            image_lab = cv2.cvtColor(self.data, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit, tileGridSize) #clipLimit=2.0, tileGridSize=(8, 8)
            image_lab[..., 0] = clahe.apply(image_lab[..., 0])
            color_equalized = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

            self.data = color_equalized

        return self