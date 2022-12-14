import numpy as np
from Histogram import *


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """

    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)

    def align_channel(self, channel: np.ndarray, tail_elimination: bool = False) -> np.ndarray:
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
            R = self.align_channel(self.data[..., 0], tail_elimination)
            G = self.align_channel(self.data[..., 1], tail_elimination)
            B = self.align_channel(self.data[..., 2], tail_elimination)

            self.data = np.dstack((R, G, B))
                # self.data[:, :, i] = self.align_channel(img_data, tail_elimination)

        if self.color_model == 4:
            self.data = self.align_channel(self.data, tail_elimination)
        return self