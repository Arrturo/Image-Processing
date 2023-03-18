from typing import Optional
from GrayScaleTransform import *
import cv2

class EdgeDetection(GrayScaleTransform):

    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        super().__init__(path, color_model)


    def canny(self, th0: int, th1: int, kernel_size: int) -> BaseImage:
        """
        Detekcja krawędzi metodą Canny'ego
        jest rozwinięciem operatora Sobela.
        Działanie jej polega na wykonaniu nastepujących kroków:
        """

        canny_edges = cv2.Canny(
            self.data,
            th0,  # prog histerezy 1
            th1,  # prog histerezy 2
            kernel_size  # wielkoscc filtra sobela
        )

        self.data = canny_edges
        return self


    def hough_circles(self, rgb: tuple, method=cv2.HOUGH_GRADIENT, dp=2, minDist=60, minRadius=20, maxRadius=100):
        """
        method – metoda detekcji okręgów
        dp – parametr określający, jak bardzo obraz jest pomniejszany przed detekcją okręgów
        minDist – minimalna odległość między środkami wykrytych okręgów
        minRadius – minimalny promień okręgu
        maxRadius – maksymalny promień okręgu
        """

        org = self.data
        gray = self.to_gray_data()
        circles = cv2.HoughCircles(gray, method=method, dp=dp, minDist=minDist, minRadius=minRadius, maxRadius=maxRadius)

        for (x, y, z) in circles.astype(int)[0]:
            cv2.circle(org, (x, y), z, rgb, 4)
        
        self.data = org
        return self
    

    def hough_lines(self, image: GrayScaleTransform, rgb: tuple, rho=2, theta=(np.pi/180), threshold=30, minLineLength=0, maxLineGap=0) -> np.ndarray:
        """
        rho – odległość w pikselach, z jaką dokładnością szukamy linii
        theta – dokładność w radianach, z jaką szukamy linii
        threshold – minimalna liczba punktów, które muszą być na linii, aby została ona zwrócona
        minLineLength – minimalna długość linii, która może zostać zwrócona
        maxLineGap – maksymalna odległość między punktami, które są rozdzielone na linii, aby została ona zwrócona
        """

        lines = cv2.HoughLinesP(self.data, rho,
                                theta, threshold, minLineLength, maxLineGap)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image.data, (x1, y1), (x2, y2), rgb, 5)

        self.data = cv2.cvtColor(self.data, cv2.COLOR_GRAY2RGB)
        return self