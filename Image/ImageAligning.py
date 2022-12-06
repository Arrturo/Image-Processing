from BaseImage import *
from Histogram import *


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def align_image(self, tail_elimination: bool = False) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramów
        """

        if self.color_model == 4:
            data = self.data

            if tail_elimination == False:
                M = np.max(self.data)
                m = np.min(self.data)

            if tail_elimination == True:
                cumulated_hist = Histogram(self.data).to_cumulated()
                M = np.quantile(cumulated_hist.values, 0.05)
                m = np.quantile(cumulated_hist.values, 0.95)

            self.data = (data - m) * (255 / (M - m))

        #in progress
        if self.color_model == 0:
            R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            for layer in [R, G, B]:
                if tail_elimination == False:
                    M = np.max(layer.all())
                    m = np.min(layer.all())

                if tail_elimination == True:
                    cumulated_hist = Histogram(layer).to_cumulated()
                    M = np.quantile(cumulated_hist.values, 0.05)
                    m = np.quantile(cumulated_hist.values, 0.95)

                layer = (layer - m) * (255 / (M - m))
            self.data = np.dstack((R, G, B)).astype('uint8')
        # return __self__.class__(self.data)

        return self

