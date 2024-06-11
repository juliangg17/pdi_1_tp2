import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    else:
        plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def filtro_area(img, area_threshold, keep_larger=True):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    output = np.zeros_like(img)
    for i, stat in enumerate(stats):
        if i == 0:  # Ignorar el fondo
            continue
        ancho = stat[cv2.CC_STAT_WIDTH]
        alto = stat[cv2.CC_STAT_HEIGHT]
        area = ancho * alto
        if (keep_larger and area > area_threshold) or (not keep_larger and area < area_threshold):
            output[labels == i] = 255
    return output

def dilation(mask, kernel_size, iterations):
    # Crear el kernel para la dilatación
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 10)) #línea horizontal
    # Aplicar la dilatación
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def filtrar_por_relacion_aspecto(img, aspecto=0.5, menor=True):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    output = np.zeros_like(img)
    for i, stat in enumerate(stats):
        if i == 0:  # Ignorar el fondo
            continue
        ancho = stat[cv2.CC_STAT_WIDTH]
        alto = stat[cv2.CC_STAT_HEIGHT]
        aspect_ratio = ancho / alto
        if (menor and aspect_ratio < aspecto) or (not menor and aspect_ratio > aspecto):
            output[labels == i] = 255
    return output

def patente(archivo,p,kernel):
    imagen = cv2.imread(archivo)  # Cargar imagen en BGR
    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) 
    img = cv2.equalizeHist(img)
    if kernel:
        sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, sharp_kernel) # Aplicar filtro de nitidez
    img = cv2.add(img, 75) # aumentar brillo (segundo parámetro)
    _, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY) #umbralizar la imagen
    img = filtrar_por_relacion_aspecto(img,.30,False)
    img = filtrar_por_relacion_aspecto(img,.65,True)
    img = filtro_area(img,350,False)
    img = filtro_area(img,30,True)
    img = dilation(img, kernel_size=16, iterations=2)
    img = filtro_area(img,500,True)
    img = filtrar_por_relacion_aspecto(img,1.4,False)
    img = filtro_area(img,700,True)
    _, _, stats, _ = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S) # Etiquetado de componentes conectados -
    for st in stats:
        cv2.rectangle(imagen, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=2)
    if p==1:
        #imshow(img, title='', color_img=True)
        imshow(imagen, title='', color_img=True)
    return imagen, stats

def crop_patente(archivo,p,kernel=False):
    imagen,stats=patente(archivo,0,kernel)
    cropped_images = []
    for st in stats:
        x, y, w, h, _ = st
        cropped_image = imagen[y+5:y+h-5, x+5:x+w-5]
        cropped_images.append(cropped_image)
    if p==1:
        imshow(cropped_images[1], title=f'Cropped Image', color_img=True)
    return cropped_images[1]

def comp_patente(archivo,umbral,p,kernel=False):
    imagen=crop_patente(archivo,p,kernel)
    img = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY) #umbralizar la imagen
    img = filtrar_por_relacion_aspecto(img,.30,False)
    img = filtrar_por_relacion_aspecto(img,.7,True)
    img = filtro_area(img,30,True)
    _, _, stats, _ = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S) # Etiquetado de componentes conectados
    for st in stats:
        cv2.rectangle(imagen, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=1)    
    if p==1:
        imshow(img, title='', color_img=False)
        imshow(imagen, title='', color_img=False)
    return imagen, stats

def letras(archivo):
    max_len, best_threshold, best_kernel = 1, 0, False
    for kernel in [False, True]:
        for umbral in range(110, 151, 10):
            try:
                _, stats = comp_patente(archivo, umbral, 0, kernel)
                if 8 > len(stats) > max_len:
                    max_len, best_threshold, best_kernel = len(stats), umbral, kernel
            except IndexError:
                break
    img, stats = comp_patente(archivo, best_threshold, 0, best_kernel)
    imshow(img, title='', color_img=False)        
    return best_kernel

def detector(archivo):
    kernel=letras(archivo)
    patente(archivo,1,kernel)

detector('Patentes/img09.png')