import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_img(path):
    img = Image.open(path)
    return np.array(img)

def svd_comp(ch, k):
    U, s, Vt = np.linalg.svd(ch, full_matrices=False)
    comp = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return comp

img_path = './bocchi.jpg'
save_dir = './results'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

orig = load_img(img_path)
k_list = [1, 2, 4, 16]

chs = []
comp_imgs = []

h, w, c = orig.shape
orig_2d = orig.reshape(h * w, c)
orig_norm = np.linalg.norm(orig_2d, 'fro')

for i in range(3):
    ch = orig[:, :, i]
    comp_chs = []
    
    for k in k_list:
        comp = svd_comp(ch, k)
        comp_chs.append(comp)
    
    chs.append(comp_chs)

for idx, k in enumerate(k_list):
    comp_img = np.stack([chs[0][idx], chs[1][idx], chs[2][idx]], axis=2)
    comp_imgs.append(comp_img)
    
    comp_2d = comp_img.reshape(h * w, c)
    comp_norm = np.linalg.norm(comp_2d, 'fro')
    pct = comp_norm / orig_norm * 100
    print(f"k={k}: {pct:.2f}% F-norm")
    
    save_img = np.clip(comp_img, 0, 255).astype(np.uint8)
    save_path = os.path.join(save_dir, f'k_{k}.jpg')
    Image.fromarray(save_img).save(save_path)

plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(orig)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(np.clip(comp_imgs[0], 0, 255).astype(np.uint8))
plt.title('k=1')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.clip(comp_imgs[1], 0, 255).astype(np.uint8))
plt.title('k=2')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.clip(comp_imgs[2], 0, 255).astype(np.uint8))
plt.title('k=4')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(np.clip(comp_imgs[3], 0, 255).astype(np.uint8))
plt.title('k=16')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"All images saved to: {save_dir}")