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

Luego se segmento la señal capturada en 5 partes, con el siguiente codigo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

data = np.loadtxt("/labo fs200.txt")
t = data[0, :]
x = data[1, :]
fs = 200  # Frecuencia de muestreo [Hz]

x_rect = np.abs(x - np.mean(x))
b, a = butter(2, 2/(fs/2), btype='low')
envolvente = filtfilt(b, a, x_rect)
envolvente_norm = envolvente / np.max(envolvente)

umbral = np.mean(envolvente_norm) + 0.5*np.std(envolvente_norm)
activa = envolvente_norm > umbral
start_idx = np.where(np.diff(activa.astype(int)) == 1)[0]
end_idx = np.where(np.diff(activa.astype(int)) == -1)[0]

if len(end_idx) > 0 and end_idx[0] < start_idx[0]:
    end_idx = end_idx[1:]
if len(start_idx) > len(end_idx):
    start_idx = start_idx[:-1]

pre_ext = int(0.15 * fs)
post_ext = int(0.10 * fs)
start_idx_adj = np.clip(start_idx - pre_ext, 0, len(x)-1)
end_idx_adj   = np.clip(end_idx + post_ext, 0, len(x)-1)

contracciones = []  # Lista para almacenar las contracciones

for i, (ini, fin) in enumerate(zip(start_idx_adj, end_idx_adj)):
    seg_t = t[ini:fin]
    seg_x = x[ini:fin]
    contracciones.append({
        "tiempo": seg_t,
        "senal": seg_x,
        "indice_inicio": ini,
        "indice_fin": fin
    })

    # Graficar cada una
    plt.figure(figsize=(6,2))
    plt.plot(seg_t, seg_x)
    plt.title(f"Contracción {i+1}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.plot(t, x/np.max(np.abs(x)), label="Señal filtrada", color='blue', alpha=0.7)
plt.plot(t, envolvente_norm, label="Energía normalizada", color='orange', linewidth=2)
for i in range(len(start_idx_adj)):
    plt.axvspan(t[start_idx_adj[i]], t[end_idx_adj[i]], color='red', alpha=0.3)

plt.title("Segmentación automática de contracciones musculares (ajustada)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
evidenciandolo así en las siguientes imagenes:

<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/158493f9-7b90-4674-8bed-e65e4c01f29e" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/5a1c6de1-dad5-40d8-bcd5-a9a98cc1b50a" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/ff3ad7a4-8350-4dfe-afb0-b6c4ff9dbe58" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/0139f105-9cda-4814-8caf-339d8a645a83" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/cce05647-3072-4066-92fb-37b1b0e7cc31" />
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/dbf849fa-8c45-4efa-b116-15a21fda0673" />

Para así tomar la frecuencia media y la frecuencia mediana con el siguiente codigo:

```python
import pandas as pd
import matplotlib.pyplot as plt

df_resultados = pd.DataFrame(resultados)
df_resultados.columns = ["Contracción", "Frecuencia media (Hz)", "Frecuencia mediana (Hz)"]

# Mostrar tabla
print("\n=== TABLA DE RESULTADOS ===\n")
print(df_resultados.to_string(index=False))

plt.figure(figsize=(8,4))
plt.plot(df_resultados["Contracción"], df_resultados["Frecuencia media (Hz)"], marker='o', label="Frecuencia media", color='steelblue')
plt.plot(df_resultados["Contracción"], df_resultados["Frecuencia mediana (Hz)"], marker='s', label="Frecuencia mediana", color='orange')

plt.title("Evolución de las frecuencias por contracción")
plt.xlabel("Número de contracción")
plt.ylabel("Frecuencia [Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
obteniendo una tabla y la siguiente grafica:


#### TABLA DE RESULTADOS 
| Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz) |
|--------------|-----------------------|--------------------------|
| 1            | 534.134160            | 283.018868              |
| 2            | 524.413970            | 283.018868              |
| 3            | 546.472563            | 285.714286              |
| 4            | 543.589980            | 280.373832              |
| 5            | 540.716311            | 280.373832              |


<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/4ccb0ebd-9880-427b-9e49-8539cd6fd7ad" />

Durante la serie de contracciones simuladas, tanto la frecuencia media como la mediana muestran variaciones leves, sin una disminución sostenida. Esto indica que la señal muscular se mantiene estable, sin evidencia de fatiga progresiva a lo largo del ejercicio

## Parte B

![Diagramas lab 4_page-0002](https://github.com/user-attachments/assets/4fbf2236-846e-4194-8106-702e431986bd)

## Parte C 

![Diagramas lab 4_page-0003](https://github.com/user-attachments/assets/61307227-83f8-4225-9303-267050daeab5)


