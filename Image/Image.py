from ImageComparison import *
from ImageAligning import *


class Image(ImageComparison, ImageAligning):
    """
    klasa stanowiaca glowny interfejs biblioteki
    """
    
    def __init__(self, path: str) -> None:
        super().__init__(path)