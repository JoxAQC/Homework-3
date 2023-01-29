import albumentations as albu
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def menu():
    menu = """
    ---------HOMEWORK 3: IMAGE TRANSFORMATIONS----------
    1.Mostrar canal rojo
    2.Reducir la resolución espacial a 72dpi
    3.Seleccionar canal y cambiar de la imagen a
      16 y 2 niveles de intensidad
    4.Rotar 90°
    5.Reflejar la imagen
    6.Aplicar una mascara circular al rostro
    7.Aplicar un efecto shear
    8.Aplicar filtros del 1-7
    Elija una opcion: """
    option = int(input(menu))
    if option == 1:
        pass
    elif option ==2:
        pass
    elif option ==3:
        pass
    elif option ==4:
        image = cv2.imread("rosie.png")
        image_norm = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Rotated Image', image_norm)
        cv2.imwrite("Rotated Image.jpg", image_norm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        menu()
    elif option ==5:
        imagen = cv2.imread('ave.jpg')
        flip0 = cv2.flip(imagen,0)
        cv2.imshow('Reflected Image',flip0)
        cv2.imwrite("Reflected Image.jpg", flip0)
        cv2.waitKey(0)
        
    elif option ==6:
        pass
    elif option ==7:
        # Leer imagen
        img = cv2.imread("rosie.png")
        # Obtener las dimensiones de la imagen
        height, width = img.shape[:2]
        # Generar la matriz de transformación shear
        matrix = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
        # Aplicar el efecto shear
        img_shear = cv2.warpAffine(img, matrix, (width, height))
        # Mostrar la imagen modificada
        cv2.imshow("Shear image", img_shear)
        cv2.imwrite("Shear image.jpg", img_shear)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif option ==8:
        #Reconocimiento facial
        faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread("rosie.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceClassif.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            maxSize=(800,800))

        for (x,y,w,h) in faces:
            cv2.circle(img, (int(x + w/2), int(y + h/2)), int(h/2), (255, 0, 0), 2)


        # Obtener las dimensiones de la imagen
        height, width = img.shape[:2]

        # Generar la matriz de transformación shear
        matrix = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)

        # Aplicar el efecto shear
        img_shear = cv2.warpAffine(img, matrix, (width, height))

        #Rotar la imagen 90°
        imgR = cv2.rotate(img_shear, cv2.ROTATE_90_CLOCKWISE)

        #Reflejar la imagen
        flip0 = cv2.flip(imgR,0)

        #Reducir la resolución espacial a 72dpi
        im_ = Image.fromarray(flip0)

        im_.info['dpi'] = (72, 72)

        #Mostrar canal rojo
        im_xd = np.array(im_)

        # Separar los canales de color de la imagen original
        B, G, R = cv2.split(im_xd)

        # Crear una imagen solo con el canal rojo
        img_red = cv2.merge((R, R, R))

        # Guardar la imagen modificada en disco duro
        cv2.imwrite("canal_rojo.png", img_red)


        #convertir la imagen a un array numpy
        im_array = np.array(img_red)

        #aplicar una función de escalamiento para reducir el número de niveles de intensidad a 16
        im_array16 = (im_array // 16)*  16

        #convertir el array numpy de nuevo a una imagen PIL
        im_16 = Image.fromarray(im_array16)

        #guardar la imagen con 16 niveles de intensidad
        im_16.save("rosie_16.png")

        ##Imagen a 2 niveles de intesidad
        im2 = im_.convert('1')

        #guardar la imagen para convertir a 2 niveles de intensidad
        im2.save("rosie_2.png")

        # Mostrar imagen final
        cv2.imshow("canal_rojo.png", img_red)
        cv2.waitKey(0)
        
    else:
        print("Opcion no valida.")

if __name__ == "__main__":
    menu()


