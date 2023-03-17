from ImageComparison import *
from ImageAligning import *
from ImageFiltration import *
from Thresholding import *
from EdgeDetection import *


class Image(ImageComparison, ImageAligning, ImageFiltration, Thresholding, EdgeDetection):
    """
    klasa stanowiaca glowny interfejs biblioteki
    """

    def __init__(self, path: Union[str, np.ndarray], color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)
