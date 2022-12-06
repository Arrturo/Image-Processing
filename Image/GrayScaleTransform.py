from BaseImage import *
import numpy as np


class GrayScaleTransform(BaseImage):
    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)

    def to_gray_data(self) -> np.ndarray:
        if self.color_model == 0:
            R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            gray_img = 0.2989 * R + 0.587 * G + 0.114 * B

        if self.color_model == 4:
            gray_img = self.get_layer(0)

        return gray_img

    def to_gray(self) -> 'BaseImage':
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """

        self.data = self.to_gray_data()
        self.color_model = 4
        return self

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> 'BaseImage':
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """

        if self.color_model == 0:
            gray_img = self.to_gray().data
            sepia_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
            alpha, beta = alpha_beta

            if alpha_beta != (None, None):
                if (alpha > 1 > beta) and (alpha + beta == 2):
                    sepia_img[:, :, 0] = np.clip(alpha * gray_img, 0, 255)  # sepia_img[..., 0]
                    sepia_img[:, :, 1] = np.clip(gray_img, 0, 255)
                    sepia_img[:, :, 2] = np.clip(gray_img * beta, 0, 255)

            if w is not None:
                if 20 <= w <= 40:
                    sepia_img[:, :, 0] = np.clip(gray_img + 2 * w, 0, 255)
                    sepia_img[:, :, 1] = np.clip(gray_img + w, 0, 255)
                    sepia_img[:, :, 2] = np.clip(gray_img, 0, 255)

            self.data = sepia_img
            self.color_model = 5
            return self
