import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

###############################################################################################
#FUNCIONES
# definimos función genérica para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def erosion(mask, kernel_size, iterations):
    # Crear el kernel para la erosión
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Aplicar la erosión
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    
    return eroded_mask

def dilation(mask, kernel_size, iterations):
    # Crear el kernel para la dilatación
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Aplicar la dilatación
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def filtro_area(img, area_threshold, keep_larger=True):
    # Encontrar los contornos
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Crear una máscara en blanco
    mask = np.zeros_like(img)
    # Dibujar solo los contornos que cumplen el criterio del umbral de área
    for contour in contours:
        if (keep_larger and cv2.contourArea(contour) > area_threshold) or (not keep_larger and cv2.contourArea(contour) < area_threshold):
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    # Aplicar la máscara a la imagen original
    img = cv2.bitwise_and(img, img, mask=mask)
    return img

def filtro_forma(img, circularity_threshold, keep_higher=True):
    # Crear una máscara en blanco
    mask_circular = np.zeros_like(img)
    # Filtrar y dibujar solo los contornos que cumplen el criterio de circularidad
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter != 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if (keep_higher and circularity > circularity_threshold) or (not keep_higher and circularity < circularity_threshold):
                cv2.drawContours(mask_circular, [contour], -1, 255, thickness=cv2.FILLED)
    # Aplicar la máscara a la imagen original
    img = cv2.bitwise_and(img, img, mask=mask_circular)
    return img

def label_connected_components(img, connectivity=4):
    # Etiquetado de componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  
    # Aplicar mapa de color a la imagen de etiquetas
    im_color = cv2.applyColorMap(np.uint8(255 / num_labels * labels), cv2.COLORMAP_JET)
    # Dibujar círculos en los centroides
    for centroid in centroids:
        cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255, 255, 255), thickness=-1)
    # Dibujar rectángulos alrededor de los componentes conectados
    for st in stats:
        cv2.rectangle(im_color, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=2)  
    return im_color

#######################################################################################################################################################


#AISLAR RESISTENCIAS ##########################################################################
img = cv2.cvtColor(cv2.imread('placa.png'), cv2.COLOR_BGR2RGB)  #carga imagen
img = cv2.medianBlur(img, 17)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #convertir imagen a escala de grises
img = cv2.Canny(img, threshold1=0.02*255, threshold2=0.08*255)   #detección bordes canny
img = dilation(img, kernel_size=3, iterations=1)  #dilatar imagen
img = filtro_area(img,9000,True)
img = filtro_area(img,19000,False)
img = filtro_forma(img,0.05,True)
img = filtro_forma(img,0.35,False)
img = dilation(img, kernel_size=8, iterations=2)  #dilatar imagen
img_resi = erosion(img, kernel_size=5, iterations=2)  #dilatar imagen
imshow(img_resi, title='', color_img=False)

#AISLAR CAPACITORES ##########################################################################
img = cv2.cvtColor(cv2.imread('placa.png'), cv2.COLOR_BGR2GRAY)  #carga imagen
img = cv2.add(img, 18) # aumentar brillo (segundo parámetro)
_, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY) #umbralizar la imagen
img = filtro_area(img,5600)
img = dilation(img, kernel_size=5, iterations=1)  #dilatar imagen
img_cap = filtro_forma(img,0.4)
imshow(img_cap, title='', color_img=False)

#AISLAR CHIP ##########################################################################
img = cv2.cvtColor(cv2.imread('placa.png'), cv2.COLOR_BGR2RGB)  #carga imagen
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #convertir a hsv
lower_gray = np.array([0, 0, 46])   #límite para gris más oscuro
upper_gray = np.array([179, 50, 220])   #límite para gris más claro
mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)    #máscara para los tonos grises
gray_components = cv2.bitwise_and(img, img, mask=mask_gray) #aplicar máscara a imagen original
gray_image = cv2.cvtColor(gray_components, cv2.COLOR_RGB2GRAY)  #convertir imagen a escala de grises
_, img = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)  #binarizar imagen
img = dilation(img, kernel_size=10, iterations=1)  #dilatar imagen
img = filtro_area(img,100000,True)
img = filtro_area(img,160000,False)
img_chip = filtro_forma(img,0.2,False)
imshow(img_chip, title='', color_img=False)

# EJERCICIO A ################################################################################
imagen = cv2.cvtColor(cv2.imread('placa.png'), cv2.COLOR_BGR2RGB)  #carga imagen

num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_resi, 8, cv2.CV_32S) # Etiquetado de componentes conectados -
for st in stats: # Dibujar rectángulos alrededor de los componentes conectados
    cv2.rectangle(imagen, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 255, 0), thickness=5)  

num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_cap, 8, cv2.CV_32S) # Etiquetado de componentes conectados -
for st in stats: # Dibujar rectángulos alrededor de los componentes conectados
    cv2.rectangle(imagen, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(0, 0, 255), thickness=5)  

num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_chip, 8, cv2.CV_32S) # Etiquetado de componentes conectados -
for st in stats: # Dibujar rectángulos alrededor de los componentes conectados
    cv2.rectangle(imagen, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), color=(255, 0, 0), thickness=5)  

imshow(imagen, color_img=True, title='')

# EJERCICIO B ################################################################################
imagen = cv2.cvtColor(cv2.imread('placa.png'), cv2.COLOR_BGR2RGB)  #carga imagen
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_cap, 8, cv2.CV_32S)
areas = stats[1:, cv2.CC_STAT_AREA]  #Extraer las áreas, omitir la primera etiqueta que es el fondo
kmeans = KMeans(n_clusters=4, random_state=42).fit(areas.reshape(-1, 1)) # Aplicar K-means clustering a las áreas
cluster_labels = kmeans.labels_
colores = {'rojo': (0, 0, 255), 'verde': (0, 255, 0), 'azul': (255, 0, 0), 'amarillo': (255, 255, 0)} # Colores para los diferentes grupos
color_names = ['rojo', 'verde', 'azul', 'amarillo']

# Contador para las categorías
categoria_contador = {color: 0 for color in color_names}

# Dibujar rectángulos alrededor de los componentes conectados según su grupo
for idx, st in enumerate(stats[1:], start=1):  # Omitir la primera etiqueta que es el fondo
    color_name = color_names[cluster_labels[idx - 1]]
    categoria_contador[color_name] += 1
    color = colores[color_name]
    cv2.rectangle(imagen, (st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP]), 
                  (st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT]), 
                  color=color, thickness=5)
# Mostrar la imagen con los rectángulos dibujados
imshow(imagen, color_img=True, title='Capacitores Agrupados por Área')
for color, count in categoria_contador.items():
    print(f'Cantidad de capacitores de color {color}: {count}')

# EJERCICIO C ################################################################################
num_labels, _, _, _ = cv2.connectedComponentsWithStats(img_resi, 8, cv2.CV_32S) # Etiquetado de componentes conectados 
print('Cantidad de resistencias:', num_labels - 1)