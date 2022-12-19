from GrayScaleTransform import *


class Thresholding(GrayScaleTransform):

    def threshold(self, value: int) -> BaseImage:
        """
        metoda dokonujaca operacji segmentacji za pomoca binaryzacji
        """

        if self.color_model == 0:
            self.to_gray()

        self.data[self.data < value] = 0
        self.data[self.data >= value] = 255

        return self