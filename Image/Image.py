from ImageComparison import *
from ImageAligning import *
from ImageFiltration import *
from Thresholding import *


class Image(ImageComparison, ImageAligning, ImageFiltration, Thresholding):
    """
    klasa stanowiaca glowny interfejs biblioteki
    """
    
    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)