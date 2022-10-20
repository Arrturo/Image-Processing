from matplotlib.image import imsave, imread
from matplotlib.pyplot import imshow
from matplotlib.colors import hsv_to_rgb
from enum import Enum
import numpy as np
np.seterr(over='ignore')


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        self.data = imread(path)
        self.color_model = 0

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """
        imsave(path, self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """
        match self.color_model:
            case 0: imshow(self.data)
            case 1: imshow(hsv_to_rgb(self.data))
            # case 2: imshow(self.data)
            # case 3: imshow((self.data))

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
                    nominator = (R[i, j] - (0.5 * G[i, j]) - (0.5 * B[i, j]))
                    denominator = (R[i, j] ** 2) + (G[i, j] ** 2) + (B[i, j] ** 2) - (R[i, j] * G[i, j]) - (R[i, j] * B[i, j]) - (G[i, j] * B[i, j])
                    if G[i, j] >= B[i, j]:
                        H[i, j] = np.int16(np.arccos(nominator / np.sqrt(denominator)))
                    else:
                        H[i, j] = np.int16(360 - np.arccos(nominator / np.sqrt(denominator)))
            self.data = np.dstack((H, S, V), axis=-1)
            self.color_model = 1
            return self

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model == 0:
            R, G, B = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            
            H = np.zeros((R.shape[0], R.shape[1]))
            S = np.zeros((R.shape[0], R.shape[1]))
            I = np.zeros((R.shape[0], R.shape[1]))

            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    M = max(R[i, j].max(), G[i, j].max(), B[i, j].max())
                    m = min(R[i, j].min(), G[i, j].min(), B[i, j].min())
                    I[i, j] = (R[i, j] + G[i, j] + B[i, j]) / 3 
                    if M > 0:
                        S[i, j] = 1 - (m / M)
                    else:
                        S[i, j] = 0
                    nominator = (R[i, j] - (0.5 * G[i, j]) - (0.5 * B[i, j]))
                    denominator = (R[i, j] ** 2) + (G[i, j] ** 2) + (B[i, j] ** 2) - (R[i, j] * G[i, j]) - (R[i, j] * B[i, j]) - (G[i, j] * B[i, j])
                    if G[i, j] >= B[i, j]:
                        H[i, j] = np.arccos(nominator / np.sqrt(denominator))
                    else:
                        H[i, j] = 360 - np.arccos(nominator / np.sqrt(denominator))
            self.data = np.stack((H, S, I), axis=-1)

        self.color_model = 2
        return self.data

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

            for j in range(R.shape[0]):
                for i in range(R.shape[1]):
                    M = max(R[i, j].max(), G[i, j].max(), B[i, j].max())
                    m = min(R[i, j].min(), G[i, j].min(), B[i, j].min())
                    d = (M - m / 255)
                    L[i, j] = (0.5 * (M + m)) / 255

                    if L[i, j] > 0:
                        S[i, j] = d / (1 - 2 * L[i, j] - 1)
                    else:
                        S[i, j] = 0
                    nominator = (R[i, j] - (0.5 * G[i, j]) - (0.5 * B[i, j]))
                    denominator = (R[i, j] ** 2) + (G[i, j] ** 2) + (B[i, j] ** 2) - (R[i, j] * G[i, j]) - (R[i, j] * B[i, j]) - (G[i, j] * B[i, j])
                    if G[i, j] >= B[i, j]:
                        H[i, j] = np.int16(np.arccos(nominator / np.sqrt(denominator)))
                    else:
                        H[i, j] = np.int16(360 - np.arccos(nominator / np.sqrt(denominator)))
            self.data = np.stack((H, S, L), axis=-1)

        self.color_model = 3
        return self.data

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
                    z = (M - m) * 1 - (abs((H[i, j] / 60) % 2) - 1)

                    if 0 <= H[i, j] < 60:
                        R[i, j] = M
                        G[i, j] = z + m
                        B[i, j] = m

                    if 60 <= H[i, j] < 120:
                        R[i, j] = z + m
                        G[i, j] = M
                        B[i, j] = m

                    if 120 <= H[i, j] < 180:
                        R[i, j] = m
                        G[i, j] = M
                        B[i, j] = z + m

                    if 180 <= H[i, j] < 240:
                        R[i, j] = m
                        G[i, j] = M
                        B[i, j] = z + m

                    if 240 <= H[i, j] < 300:
                        R[i, j] = z + m
                        G[i, j] = m
                        B[i, j] = M

                    if 300 <= H[i, j] < 360:
                        R[i, j] = M
                        G[i, j] = m
                        B[i, j] = z + m
            self.data = np.dstack((R, G, B)).astype('uint16')

        if self.color_model == 2:
            H, S, I = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            R = np.zeros((H.shape[0], H.shape[1]))
            G = np.zeros((H.shape[0], H.shape[1]))
            B = np.zeros((H.shape[0], H.shape[1]))

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):

                    if H[i, j] == 0:
                        R[i, j] = round(I[i, j] + 2 * I[i, j] * S[i, j])

                    if 0 < H[i, j] < 120:
                        R[i, j] = round(I[i, j] + I[i, j] * S[i, j] * np.cos(H[i, j]) / np.cos(60 - H[i, j]))
                        G[i, j] = round(I[i, j] + I[i, j] * S[i, j] * (1 - np.cos(H[i, j]) / np.cos(60 - H[i, j])))
                        I[i, j] = round(I[i, j] * S[i, j])

                    if H[i, j] == 120:
                        R[i, j] = round(I[i, j] - I[i, j] * S[i, j])
                        G[i, j] = round(I[i, j] + 2 * I[i, j] * S[i, j])
                        B[i, j] = round(I[i, j] - I[i, j] * S[i, j])

                    if 120 < H[i, j] < 240:
                        G[i, j] = round(I[i, j] + I[i, j] * S[i, j] * np.cos(H[i, j] - 120) / np.cos(180 - H[i, j]))
                        B[i, j] = round(I[i, j] + I[i, j] * S[i, j] * (1 - np.cos(H[i, j] - 120) / np.cos(180 - H[i, j])))
                        R[i, j] = round(I[i, j] - I[i, j] * S[i, j])

                    if H[i, j] == 240:
                        R[i, j] = round(I[i, j] - I[i, j] * S[i, j])
                        G[i, j] = round(I[i, j] - I[i, j] * S[i, j])
                        B[i, j] = round(I[i, j] + 2 * I[i, j] * S[i, j])

                    if 240 < H[i, j] < 360:
                        R[i, j] = round(I[i, j] + I[i, j] * S[i, j] * (np.cos(H[i, j] - 240) / np.cos(300 - H[i, j])))
                        G[i, j] = round(I[i, j] - I[i, j] * S[i, j])
                        B[i, j] = round(I[i, j] + I[i, j] * S[i, j] * (np.cos(H[i, j] - 240) / np.cos(300 - H[i, j])))
                self.data = np.dstack((R, G, B)).astype('uint16')

        if self.color_model == 3:
            H, S, L = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            R = np.zeros((H.shape[0], H.shape[1]))
            G = np.zeros((H.shape[0], H.shape[1]))
            B = np.zeros((H.shape[0], H.shape[1]))

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    d = S[i, j] * (1 - abs(2 * L[i, j] - 1))
                    m = 255 * (L[i, j] - (0.5 * d))
                    x = d * (1 - abs(H[i, j] / 60 % 2 - 1))

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
