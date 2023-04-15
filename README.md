# Machine Graphics
This repository contains Python implementations of various operations on images:
- [HSV, HSI, HSL convertion](https://github.com/Arrturo/Machine-Graphics/blob/main/lab2/main.ipynb)
- [Sepia, Grayscale image conversion](https://github.com/Arrturo/Machine-Graphics/blob/main/lab3/main.ipynb)
- [image histogram](https://github.com/Arrturo/Machine-Graphics/blob/main/lab4/main.ipynb)
- [*MSE, RMSE image comparison*](https://github.com/Arrturo/Machine-Graphics/blob/main/lab4/main.ipynb)
- [*aligning image with tail elimination*](https://github.com/Arrturo/Machine-Graphics/blob/main/lab5/main.ipynb)
- [*convolution*](https://github.com/Arrturo/Machine-Graphics/blob/main/lab6/main.ipynb)
- [*treshold*](https://github.com/Arrturo/Machine-Graphics/blob/main/lab7/main.ipynb)
- [*edge and circle detection*](https://github.com/Arrturo/Machine-Graphics/blob/main/lab8/lab8.ipynb)

## Table of contents
* [General info](#machine-graphics)
* [Technologies](#technologies)
* [Get Started](#get-started)
* [Usage](#usage)

## Technologies:
1. Python 3.10
2. NumPy
3. Matplotlib
4. OpenCV

## Get Started:
1. Clone the repository to your local machine:
```
git clone https://github.com/Arrturo/Machine-Graphics.git
```
2. Install necessary libraries, you can use pip:
```
pip install numpy
pip install matplotlib
pip install opencv-python
```
You can also install all needed libraries with ```requirements.txt``` file:
```
pip install -r requirements.txt
```

## Usage
To use repo, you first need to import the Image class:
```python
from Image import Image
```
Then you can create an instance of the Image class and pass in path to the image:
```python
image = Image('D://Dokumenty//Code//Python//Introduction-to-Computer-Graphics//data//lena.jpg')
```
You can then display image using the ```show_img``` method:
```python
image.show_img()
```
![](https://github.com/Arrturo/Machine-Graphics/blob/main/data/output.png)
