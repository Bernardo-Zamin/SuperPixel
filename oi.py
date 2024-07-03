import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

# Carregar a imagem
image_path = '/home/zamin/Downloads/T2-IVC/image.jpg'
image = io.imread(image_path)

# Aplicar SLIC para segmentação
segments = slic(image, n_segments=250, compactness=10, sigma=1, start_label=1)

# Marcar os limites dos segmentos
image_with_boundaries = mark_boundaries(image, segments)

# Função para capturar cliques
clicks = []

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicks.append((x, y))
        plt.scatter(x, y, c='red', s=10)
        plt.draw()

# Exibir a imagem segmentada e configurar o callback de clique
fig, ax = plt.subplots()
ax.imshow(image_with_boundaries)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Clique nas regiões segmentadas")
plt.axis('off')
plt.show()

# Exibir os cliques capturados
print("Coordenadas dos cliques:", clicks)

# Criar uma máscara com as regiões selecionadas
mask = np.zeros(segments.shape, dtype=np.bool)
for click in clicks:
    x, y = click
    segment_id = segments[y, x]
    mask[segments == segment_id] = True

# Exibir a máscara
plt.figure(figsize=(10, 10))
plt.imshow(mask, cmap='gray')
plt.title("Máscara das Regiões Selecionadas")
plt.axis('off')
plt.show()
