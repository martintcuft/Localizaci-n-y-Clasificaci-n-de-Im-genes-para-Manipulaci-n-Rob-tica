# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Carct_img(imagen):
    img = cv2.imread(imagen)
    #Convert BGR to RGB 
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Convert GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #5,5
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    # Display the results
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray)
    plt.title('Original Image gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def esquinas_Harrrisi(imgen):
    img = cv2.imread(imgen)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect corners using the Harris method
    dst = cv2.cornerHarris(gray, 3, 5, 0.1)
    # Create a boolean bitmap of corner positions
    corners = dst > 0.10 * dst.max()
    # Find the coordinates from the boolean bitmap
    coord = np.argwhere(corners)
    # Draw circles on the coordinates to mark the corners
    i= 0
    for y, x in coord:
        cv2.circle(img, (x,y), 3, (0,0,255), -1)
        print(f"Coordenadas = {x},{y} , {i}")
        i+=1
    
    # Display the image with corners
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def bordes(imgen):
    img = cv2.imread(imgen)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny method
    edges = cv2.Canny(gray, 150, 300)

    # Display the image with corners
    img[edges == 255] = (255,0,0)
    cv2.imshow('Canny Edges', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

def walala(imagen):
    img = cv2.imread(imagen)
    img_original = img.copy()

    # Preprocesamiento (Grises y desenfoque para reducir ruido)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)

    #Detección de Bordes (Canny)
    bordes = cv2.Canny(suave, 50, 150)

    #Detección de Esquinas (Harris)
    # Operamos sobre la imagen en gris (debe ser tipo float32)
    gris_float = np.float32(gris)
    dest_esquinas = cv2.cornerHarris(gris_float, blockSize=2, ksize=3, k=0.04)
    # Dilatamos el resultado para que las esquinas se vean más grandes al dibujar
    dest_esquinas = cv2.dilate(dest_esquinas, None)
    # Marcamos las esquinas en la imagen original en color ROJO
    img[dest_esquinas > 0.01 * dest_esquinas.max()] = [0, 0, 255]

    # Contornos para medir Ancho y Alto
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        if cv2.contourArea(c) < 100:
            continue
            
        # Obtener el rectángulo delimitador (Bounding Box)
        x, y, ancho, alto = cv2.boundingRect(c)
        # (Ancho x Alto) en la imagen en color AZUL
        
        texto = f"{ancho}x{alto} px"

        cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if(ancho == 383 and alto == 389):
            print("Cuadrado")
            print(x, y, ancho, alto)
            # Dibujar el rectángulo delimitador en color VERDE
            cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        if(ancho == 112 and alto == 109):
            print("circulo")
            print(x, y, ancho, alto)
            cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)


    # OpenCV usa BGR, Matplotlib usa RGB; por eso convertimos los colores
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')  # Oculta los ejes de coordenadas
    plt.show()



imagen = "C:\\Users\\Alumno\\Documents\\GitHub\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\\dataset\\Test_1\\WIN_20260702_17_07_52_Pro.jpg"

Carct_img(imagen)
esquinas_Harrrisi(imagen)
bordes(imagen)
walala(imagen)

# ruta ; Alto ; LARGO ; x, Y ; ROTACION ;  CLASE 