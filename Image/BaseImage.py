from typing import Optional
from ColorModel import *
from matplotlib.image import imsave, imread
from matplotlib.pyplot import imshow, show, subplots, tight_layout
import numpy as np
np.seterr(over='ignore')


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str, color_model: Optional[ColorModel] = 0) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """

        self.data = imread(path)
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """

        if self.color_model == 4:
            imsave(path, self.data, cmap='gray')
        else:
            imsave(path, self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """

        imshow(self.data)
        if self.color_model == 4:
            imshow(self.data, cmap='gray')

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """

        return self.data[:, :, layer_id]

    def to_hsv(self):
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model == 0:
            R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1])) 
            
            H = np.zeros((R.shape[0], R.shape[1]))
            S = np.zeros((R.shape[0], R.shape[1]))
            V = np.zeros((R.shape[0], R.shape[1]))

            for j in range(R.shape[0]):
                for i in range(R.shape[1]):

                    M = max(R[i, j], G[i, j], B[i, j])
                    m = min(R[i, j], G[i, j], B[i, j])
                    V[i, j] = M / 255

                    if M > 0:
                        S[i, j] = 1 - (m / M)
                    else:
                        S[i, j] = 0

                    nominator = R[i, j] - (0.5 * G[i, j]) - (0.5 * B[i, j])
                    denominator = (R[i, j] ** 2) + (G[i, j] ** 2) + (B[i, j] ** 2) - (R[i, j] * G[i, j]) - (R[i, j] * B[i, j]) - (G[i, j] * B[i, j])
                    if G[i, j] >= B[i, j]:
                        H[i, j] = np.arccos(nominator / (np.sqrt(denominator) + 0.0000001))
                    else:
                        H[i, j] = 360 - np.arccos(nominator / (np.sqrt(denominator) + 0.0000001))

            self.data = np.dstack((H, S, V))
            self.color_model = 1
            return self

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model == 0:
            R, G, B = np.float32(np.squeeze(np.dsplit(self.data, self.data.shape[-1])))
            
            H = np.zeros((R.shape[0], R.shape[1]))
            S = np.zeros((R.shape[0], R.shape[1]))
            I = np.zeros((R.shape[0], R.shape[1]))

            for i in range(R.shape[0]):
                for j in range(R.shape[1]):

                    m = min(R[i, j], G[i, j], B[i, j])

                    H[i][j] = 0.5 * ((R[i][j] - G[i][j]) + (R[i][j] - B[i][j])) / np.sqrt((R[i][j] - G[i][j]) ** 2 + ((R[i][j] - B[i][j]) * (G[i][j] - B[i][j]) + 0.0000001))
                    H[i][j] = np.arccos(H[i][j]) 

                    if B[i][j] <= G[i][j]:
                        H[i][j] = H[i][j]
                    else:
                        H[i][j] = (360 * np.pi) / (180 - H[i][j])

                    S[i, j] = 1 - (3 * (m / (R[i, j] + G[i, j] + B[i, j] + 0.0000001)))
                    I[i, j] = (R[i, j] + G[i, j] + B[i, j]) / 3.0

            self.data = np.dstack((H, S, I))
            self.color_model = 2
            return self

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model == 0:
            R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            
            H = np.zeros((R.shape[0], R.shape[1]))
            S = np.zeros((R.shape[0], R.shape[1]))
            L = np.zeros((R.shape[0], R.shape[1]))

            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    
                    M = max(R[i, j], G[i, j], B[i, j])
                    m = min(R[i, j], G[i, j], B[i, j])
                    d = (M - m) / 255
                    L[i, j] = 0.5 * (int(M) + int(m)) / 255

                    if L[i, j] > 0:
                        S[i, j] = d / (1 - abs(2 * L[i, j] - 1))
                    else:
                        S[i, j] = 0
                        
                    nominator = (int(R[i, j]) - (0.5 * int(G[i, j])) - (0.5 * int(B[i, j])))
                    denominator = (int(R[i, j]) ** 2) + (int(G[i, j]) ** 2) + (int(B[i, j]) ** 2) - (int(R[i, j]) * int(G[i, j])) - (int(R[i, j]) * int(B[i, j])) - (int(G[i, j]) * int(B[i, j]))
                    if G[i, j] >= B[i, j]:
                        H[i, j] = np.arccos(nominator / (np.sqrt(denominator) + 0.0000001)) * 180 / np.pi 
                    if B[i, j] > G[i, j]:
                        H[i, j] = 360 - np.arccos(nominator / (np.sqrt(denominator) + 0.0000001)) * 180 / np.pi 

            self.data = np.dstack((H, S, L))
            self.color_model = 3
            return self

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """

        if self.color_model == 0:
            raise "Image is already in RGB model"

        if self.color_model == 1:
            H, S, V = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            R = np.zeros((H.shape[0], H.shape[1]))
            G = np.zeros((H.shape[0], H.shape[1]))
            B = np.zeros((H.shape[0], H.shape[1]))

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    
                    M = 255 * V[i, j]
                    m = M * (1 - S[i, j])
                    z = (M - m) * (1 - abs(((H[i, j] / 60) % 2) - 1))

                    if 0 <= H[i, j] < 60:
                        R[i, j] = M
                        G[i, j] = z + m
                        B[i, j] = m

                    elif 60 <= H[i, j] < 120:
                        R[i, j] = z + m
                        G[i, j] = M
                        B[i, j] = m

                    elif 120 <= H[i, j] < 180:
                        R[i, j] = m
                        G[i, j] = M
                        B[i, j] = z + m

                    elif 180 <= H[i, j] < 240:
                        R[i, j] = m 
                        G[i, j] = z + m
                        B[i, j] = M

                    elif 240 <= H[i, j] < 300:
                        R[i, j] = z + m 
                        G[i, j] = m
                        B[i, j] = M

                    elif 300 <= H[i, j] < 360:
                        R[i, j] = M
                        G[i, j] = m
                        B[i, j] = z + m

        if self.color_model == 2:
            H, S, I = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            R = np.zeros((H.shape[0], H.shape[1]))
            G = np.zeros((H.shape[0], H.shape[1]))
            B = np.zeros((H.shape[0], H.shape[1]))

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):

                    if B[i, j] == G[i, j] == R[i, j]:
                        H[i, j] = 0

                    if 0 <= H[i, j] <= 120:
                        B[i, j] = I[i, j] * (1 - S[i, j])
                        R[i, j] = I[i, j] * (1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - H[i, j]))
                        G[i, j] = 3 * I[i, j] - (R[i, j] + B[i, j])

                    if 120 < H[i, j] <= 240:
                        H[i, j] -= 120
                        R[i, j] = I[i, j] * (1 - S[i, j])
                        G[i, j] = I[i, j] * (1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - H[i, j]))
                        B[i, j] = 3 * I[i, j] - (R[i, j] + G[i, j])

                    if 0 < H[i, j] <= 360:
                        H[i, j] -= 240
                        G[i, j] = I[i, j] * (1 - S[i, j])
                        B[i, j] = I[i, j] * (1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - H[i, j]))
                        R[i, j] = 3 * I[i, j] - (G[i, j] + B[i, j])

            R[R > 255] = 255
            G[G > 255] = 255
            B[B > 255] = 255

        if self.color_model == 3:
            H, S, L = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            R = np.zeros((H.shape[0], H.shape[1]))
            G = np.zeros((H.shape[0], H.shape[1]))
            B = np.zeros((H.shape[0], H.shape[1]))

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):

                    d = S[i, j] * (1 - abs((2 * L[i, j]) - 1))
                    m = 255 * (L[i, j] - (0.5 * d))
                    x = d * (1 - abs(((H[i, j] / 60) % 2) - 1))

                    if 0 <= H[i, j] < 60:
                        R[i, j] = (255 * d) + m
                        G[i, j] = (255 * x) + m
                        B[i, j] = m

                    if 60 <= H[i, j] < 120:
                        R[i, j] = (255 * x) + m
                        G[i, j] = (255 * d) + m
                        B[i, j] = m

                    if 120 <= H[i, j] < 180:
                        R[i, j] = m
                        G[i, j] = (255 * d) + m
                        B[i, j] = (255 * x) + m

                    if 180 <= H[i, j] < 240:
                        R[i, j] = m
                        G[i, j] = (255 * x) + m
                        B[i, j] = (255 * d) + m

                    if 240 <= H[i, j] < 300:
                        R[i, j] = (255 * x) + m
                        G[i, j] = m
                        B[i, j] = (255 * d) + m

                    if 300 <= H[i, j] < 360:
                        R[i, j] = (255 * d) + m
                        G[i, j] = m
                        B[i, j] = (255 * x) + m

        self.data = np.dstack((R, G, B)).astype('uint16')
        self.color_model = 0
        return self      

def show_3img(photo1: 'BaseImage', photo2: 'BaseImage', photo3: 'BaseImage') -> None:
    f, photo = subplots(1, 3, figsize=(10, 10))
    photo[0].imshow(photo1.data)
    photo[1].imshow(photo2.data)
    photo[2].imshow(photo3.data)
    tight_layout()
    show()