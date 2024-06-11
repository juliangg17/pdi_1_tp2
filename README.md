# pdi_1_tp2
Trabajo Práctico 2 de Procesamiento de Imágenes 1 TUIA
####### Detección y Clasificación de Componentes Electrónicos en una Placa ################################################################################

Este proyecto contiene código en Python para detectar y clasificar componentes electrónicos (resistencias, capacitores y chips) en imágenes de una placa de circuito impreso utilizando OpenCV. El proceso incluye varias técnicas de procesamiento de imágenes como filtrado, umbralización, dilatación, erosión, análisis de componentes conectados y clustering.

## Requisitos Previos

Asegúrate de tener Python instalado en tu máquina. Puedes descargarlo desde [python.org](https://www.python.org/).

### Librerías Requeridas

Las siguientes librerías de Python son necesarias para ejecutar el código:

- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Puedes instalar estas librerías utilizando pip:
pip install opencv-python-headless numpy matplotlib scikit-learn

Archivos
tp2 ej1_pdi.py: Contiene todas las funciones necesarias para la detección y clasificación de componentes electrónicos.
placa.png: Imagen de una placa de circuito impreso para probar la detección y clasificación de componentes.

Ejecución
Para ejecutar el código, sigue estos pasos:

Asegúrate de tener todas las librerías instaladas.
Coloca la imagen placa.png en el mismo directorio que el script de Python.
Ejecuta el código de tp2 ej1_pdi.py en o un entorno de desarrollo:

####### Detección de Patentes ##################################################################################################3

Este proyecto contiene código en Python para detectar patentes en imágenes utilizando OpenCV. El proceso de detección incluye varias técnicas de procesamiento de imágenes como filtrado, umbralización, dilatación y análisis de componentes conectados.

## Requisitos Previos
Asegúrate de tener Python instalado en tu máquina. Puedes descargarlo desde [python.org](https://www.python.org/).

### Librerías Requeridas
Las siguientes librerías de Python son necesarias para ejecutar el código:
- OpenCV
- NumPy
- Matplotlib

Puedes instalar estas librerías utilizando pip:
pip install opencv-python-headless numpy matplotlib

Archivos
tp2 ej2_pdi.py: Contiene todas las funciones necesarias para la detección de patentes.
Patentes/: Una carpeta que contiene imágenes para probar la detección de patentes.

Ejecución
Para ejecutar el código, sigue estos pasos:
Asegúrate de tener todas las librerías instaladas.
Coloca las imágenes que deseas procesar en la carpeta Patentes/.
Ejecuta la función detector(archivo) en el código tp2 ej2_pdi.py en un entorno de desarrollo.
