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

En esta segunda parte se realizo la captura de las contracciones a tiepo real con ayuda de un modulo AD8232 y con electrodos, estoss se conectan al ante brazo y con la DAQ se realiza la captura. 

![Imagen de WhatsApp 2025-10-23 a las 09 32 57_ef6c2349](https://github.com/user-attachments/assets/abfbc1ac-84cf-4a14-a30c-59f939104f1c)

con ayuda del siguiente codigo se pudo hacer la captura:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nidaqmx
from nidaqmx.constants import AcquisitionType
from threading import Thread, Event
from collections import deque
import datetime
import time

fs = 10000            # Frecuencia de muestreo (Hz)
canal = "Dev5/ai0"    # Cambia según tu dispositivo
tamano_bloque = int(fs * 0.05)  # 50 ms por bloque
ventana_tiempo = 3.0             # segundos visibles en la gráfica

# Buffers
buffer_graf = deque(maxlen=int(fs * ventana_tiempo))  # solo últimos 3 s
datos_guardados = []  # toda la señal

# Control de hilos
adquiriendo = Event()
detener_hilo = Event()
thread_lectura = None


def hilo_lectura():
    """Lee continuamente datos del DAQ en un hilo aparte."""
    global datos_guardados, buffer_graf
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(canal)
    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print(f"\n▶ Adquisición iniciada en {canal} ({fs} Hz).")

    while not detener_hilo.is_set():
        if adquiriendo.is_set():
            try:
                datos = task.read(number_of_samples_per_channel=tamano_bloque)
                buffer_graf.extend(datos)
                datos_guardados.extend(datos)
            except Exception as e:
                print("⚠ Error de lectura:", e)
                break
        else:
            time.sleep(0.05)

    task.stop()
    task.close()
    print("⏹ Adquisición detenida correctamente.")


def iniciar(event):
    global thread_lectura
    if not adquiriendo.is_set():
        if thread_lectura is None or not thread_lectura.is_alive():
            detener_hilo.clear()
            thread_lectura = Thread(target=hilo_lectura, daemon=True)
            thread_lectura.start()
        adquiriendo.set()
        print("▶ Grabando...")

def detener(event):
    """Detiene y guarda los datos."""
    adquiriendo.clear()
    detener_hilo.set()
    time.sleep(0.3)

    if datos_guardados:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"senal_EMG_{timestamp}.txt"
        tiempos = np.arange(len(datos_guardados)) / fs
        data = np.column_stack((tiempos, datos_guardados))
        np.savetxt(nombre_archivo, data, fmt="%.6f", header="Tiempo(s)\tVoltaje(V)")
        print(f"✅ Señal guardada en {nombre_archivo} ({len(datos_guardados)} muestras)")
    else:
        print("⚠ No se capturaron datos.")


fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
linea, = ax.plot([], [], lw=1.2, color='royalblue')
ax.set_xlim(0, ventana_tiempo)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Voltaje [V]")
ax.set_title("Señal EMG continua en tiempo real")
ax.grid(True, linestyle="--", alpha=0.6)

x = np.linspace(0, ventana_tiempo, int(fs * ventana_tiempo))
y = np.zeros_like(x)

def actualizar(frame):
    if len(buffer_graf) > 0:
        y = np.array(buffer_graf)
        if len(y) < len(x):
            y = np.pad(y, (len(x)-len(y), 0), constant_values=0)
        linea.set_data(x, y)
    return linea,

ax_iniciar = plt.axes([0.3, 0.1, 0.15, 0.075])
ax_detener = plt.axes([0.55, 0.1, 0.2, 0.075])
btn_iniciar = Button(ax_iniciar, 'Iniciar', color='lightgreen', hovercolor='green')
btn_detener = Button(ax_detener, 'Detener y Guardar', color='lightcoral', hovercolor='red')
btn_iniciar.on_clicked(iniciar)
btn_detener.on_clicked(detener)

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, actualizar, interval=50, blit=True)
plt.tight_layout()
plt.show()
```

Luego se pasaraon los datos para graficarlos y se tomaron los primeros 10 segundos y los ultimos 10 para poder evidenciar la fatiga en la siguiente grafica con ayuda del cofigo:

```python
import numpy as np
import matplotlib.pyplot as plt

ruta_txt = "/senal_EMG_captura_2.txt"   # Cambia por el nombre de tu archivo
col_tiempo = 0            # índice de la columna de tiempo
col_voltaje = 1           # índice de la columna de voltaje


# Carga el archivo ignorando líneas vacías o comentarios
datos = np.loadtxt(ruta_txt)

# Separa las columnas
tiempo = datos[:, col_tiempo]
voltaje = datos[:, col_voltaje]


mascara_inicio = tiempo <= 10
t_inicio = tiempo[mascara_inicio]
v_inicio = voltaje[mascara_inicio]

t_final_max = tiempo.max()
mascara_final = tiempo >= (t_final_max - 10)
t_final = tiempo[mascara_final]
v_final = voltaje[mascara_final]


plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(t_inicio, v_inicio, color='b')
plt.title("Primeros 10 segundos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_final, v_final, color='r')
plt.title("Últimos 10 segundos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.tight_layout()
plt.show()
```

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/72288403-010d-46ac-a253-9b0bcf061ecd" />

Despues se aplico un filtro pasabanda (20–450 Hz) para eliminar ruido y artefactos. 

## Parte C 

![Diagramas lab 4_page-0003](https://github.com/user-attachments/assets/61307227-83f8-4225-9303-267050daeab5)


