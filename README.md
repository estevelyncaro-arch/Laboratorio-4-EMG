# Laboratorio-4-EMG
## Resumen
En el desarrollo de este laboratorio se busca analizar señales electromiográficas (EMG) emuladas por el generador de señales y señales reales, realizando una comparación de el comportamiento de cada una. En la señal de EMG real se busca detectar la fatiga muscular capturando la señal en tiempo real y aplicando filtros para elimminar el ruido para un mejor analisis de las contracciónes.
## Parte A 
![Diagramas lab 4_page-0001](https://github.com/user-attachments/assets/fac9d933-47e5-4971-a120-a657c7122291)

Para esta primera sección se realiza la captura de una señal electromiográfica (EMG) emulada por el generador de señales con ayuda de un DAQ, se simulan 5 contracciones. Una vez adquirida la señal se importa a python y se grafica implementando el siguiente código:
```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo (2 filas: tiempo y voltaje)
data = np.loadtxt("labo fs200.txt")

# Separar filas
tiempo = data[0, :]   # primera fila
senal  = data[1, :]   # segunda fila

# Graficar
plt.figure(figsize=(9, 4))
plt.plot(tiempo, senal, linewidth=1)
plt.title("Señal adquirida en Dev5/ai0")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.tight_layout()
plt.show()
```
Obteniendo el siguiente gráfico:
<img width="889" height="390" alt="image" src="https://github.com/user-attachments/assets/1cc61e34-39ba-43b6-be91-cef91040399d" />


## Parte B
![Diagramas lab 4_page-0002](https://github.com/user-attachments/assets/4fbf2236-846e-4194-8106-702e431986bd)

## Parte C 

![Diagramas lab 4_page-0003](https://github.com/user-attachments/assets/61307227-83f8-4225-9303-267050daeab5)


