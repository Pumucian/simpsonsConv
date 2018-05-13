from PIL import Image
import os, cv2
import numpy as np
import matplotlib.pyplot as plt


carpeta = "test/"
numero = "4"

path1 = "C:/Users/fali0/Desktop/Universidad/3º/FSI/practica_fsi/data/" + carpeta + numero
path2 = "C:/Users/fali0/Desktop/Universidad/3º/FSI/practica_fsi/8data/" + carpeta + numero
size = [80,140]
listing = os.listdir(path1)
for file in listing:
    im2 = plt.imread(path1 + "/" + file).astype(np.uint8)
    im2 = cv2.resize(im2, dsize=(80, 140), interpolation=cv2.INTER_CUBIC)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(im2)
    im.save(path2 + "/" + file, "JPEG")