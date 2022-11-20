from BaseImage import *
from Histogram import *

class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """

    def __init__(self, values: np.ndarray) -> None:
        """
        inicjalizator ...
        """
        
        super().__init__(values)
        self.histogram = Histogram(self.data)


    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramów
        """
        
        if self.color_model == 4:
            pass