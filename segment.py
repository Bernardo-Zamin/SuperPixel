import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.io import imsave

# Carregar a imagem e convertê-la para float
img_path = 'images/L4A0052.jpg'  # Substitua pelo caminho da sua imagem
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = img_as_float(image)

# Segmentar a imagem em superpixels
segments = slic(image, n_segments=100, compactness=10, start_label=1)

# Função para capturar cliques e destacar superpixels
clicks = []
selected_segments = []

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        segment_id = segments[y, x]
        if segment_id not in selected_segments:
            selected_segments.append(segment_id)
        clicks.append((x, y))
        update_display()

def update_display():
    highlighted_image = image.copy()
    for segment_id in selected_segments:
        highlighted_image[segments == segment_id] = [1, 0, 0]  # Destaque em vermelho
    ax.clear()
    ax.imshow(mark_boundaries(highlighted_image, segments))
    plt.axis('off')
    plt.title("Clique nas regiões segmentadas")
    plt.draw()

def save_mask():
    mask = np.zeros(segments.shape, dtype=np.uint8)
    for segment_id in selected_segments:
        mask[segments == segment_id] = 1
    mask = mask * 255  # Converter a máscara binária para formato visível
    mask_path = 'mask.png'
    imsave(mask_path, mask)
    print(f'Máscara salva em: {mask_path}')

# Exibir a imagem e configurar o callback de clique
fig, ax = plt.subplots()
ax.imshow(mark_boundaries(image, segments))
plt.axis('off')
fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Clique nas regiões segmentadas")
plt.show()

# Exibir os cliques capturados
print("Coordenadas dos cliques:", clicks)

# Salvar a máscara segmentada
save_mask()
