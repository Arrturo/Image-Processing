from ImageComparison import *
from GrayScaleTransform import *

class Image(GrayScaleTransform, ImageComparison):
    """
    klasa stanowiaca glowny interfejs biblioteki
    """
    
    def __init__(self, path: str) -> None:
        super().__init__(path)