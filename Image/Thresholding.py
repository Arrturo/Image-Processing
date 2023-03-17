from GrayScaleTransform import *
import cv2


class Thresholding(GrayScaleTransform):

    def threshold(self, value: int) -> BaseImage:
        """
        metoda dokonujaca operacji segmentacji za pomoca binaryzacji (globalna)
        """

        if self.color_model == 0:
            self.to_gray()

        self.data[self.data < value] = 0
        self.data[self.data >= value] = 255

        return self


    def otsu(self) -> BaseImage:
        """
        metoda progowania globalnego z automatycznym doborem wartości progu na podstawie rozkładu barw obrazu
        """
        if self.color_model == 0:
            self.to_gray()

        _, thresh_otsu = cv2.threshold(
            self.data,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        self.data = thresh_otsu
        return self


    def adaptive_threshold(self, block_size: int, c: int) -> BaseImage:
        """
        metoda progująca lokalne obszary całego obrazu na podstawie lokalnie dobranych wartości progowych
        """
        if self.color_model == 0:
            self.to_gray()

        thresh_adaptive = cv2.adaptiveThreshold(
            self.data,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c
        )

        self.data = thresh_adaptive
        return self
