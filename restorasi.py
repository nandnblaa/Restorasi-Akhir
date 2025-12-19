import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ================================
#            NOISE
# ================================

def add_gaussian(img, std):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_salt_pepper_gray(img, p):
    noisy = img.copy()
    h, w = img.shape
    n = int(p * h * w)
    for _ in range(n):
        noisy[random.randint(0,h-1), random.randint(0,w-1)] = 255
        noisy[random.randint(0,h-1), random.randint(0,w-1)] = 0
    return noisy

def add_salt_pepper_rgb(img, p):
    noisy = img.copy()
    h, w, _ = img.shape
    n = int(p * h * w)
    for _ in range(n):
        noisy[random.randint(0,h-1), random.randint(0,w-1)] = [255,255,255]
        noisy[random.randint(0,h-1), random.randint(0,w-1)] = [0,0,0]
    return noisy

# ================================
#        FAST FILTER RGB & GRAY
# ================================

def min_filter(img):
    return cv2.erode(img, np.ones((3,3), np.uint8))

def max_filter(img):
    return cv2.dilate(img, np.ones((3,3), np.uint8))

def mean_filter(img):
    return cv2.blur(img, (3,3))

def median_filter(img):
    return cv2.medianBlur(img, 3)

filters = {
    "Min Filter": min_filter,
    "Max Filter": max_filter,
    "Mean Filter": mean_filter,
    "Median Filter": median_filter
}

# ================================
#              MSE
# ================================

def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float))**2)

# ================================
#              MAIN
# ================================

rgb = cv2.imread("senja.jpeg")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# === TAMPIL ASLI ===
plt.figure(figsize=(8,3))
plt.subplot(1,2,1); plt.imshow(rgb); plt.title("Citra Asli RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(gray,cmap='gray'); plt.title("Citra Grayscale"); plt.axis("off")
plt.tight_layout(); plt.show()

mse_data = []

# ================================
#     NOISE + FILTER (LENGKAP)
# ================================

for noise_type in ["Gaussian", "Salt & Pepper"]:
    levels = [0.02, 0.05]

    for level in levels:

        if noise_type == "Gaussian":
            std = int(level * 255)
            rgb_n = add_gaussian(rgb, std)
            gray_n = add_gaussian(gray, std)
        else:
            rgb_n = add_salt_pepper_rgb(rgb, level)
            gray_n = add_salt_pepper_gray(gray, level)

        # === TAMPIL NOISE ===
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1); plt.imshow(rgb_n); plt.title(f"RGB {noise_type} {level}"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(gray_n,cmap='gray'); plt.title(f"Gray {noise_type} {level}"); plt.axis("off")
        plt.tight_layout(); plt.show()

        for fname, ffunc in filters.items():

            # FILTER RGB & GRAY
            rgb_f = ffunc(rgb_n)
            gray_f = ffunc(gray_n)

            # MSE (GRAYSCALE)
            mse_data.append([noise_type, level, fname, mse(gray, gray_f)])

            # === TAMPIL HASIL FILTER (RGB & GRAY) ===
            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.imshow(rgb_f)
            plt.title(f"RGB + {fname}")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.imshow(gray_f, cmap='gray')
            plt.title(f"Grayscale + {fname}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

# ================================
#        TABEL MSE
# ================================

fig, ax = plt.subplots(figsize=(12, len(mse_data)*0.35))
ax.axis('off')

table = ax.table(
    cellText=[[a,b,c,f"{d:.2f}"] for a,b,c,d in mse_data],
    colLabels=["Noise","Level","Filter","MSE (Gray)"],
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1,1.4)
plt.title("Tabel Perbandingan MSE", pad=20)
plt.show()

# ================================
#       GRAFIK BATANG MSE
# ================================

labels = [f"{a}-{c}-{b}" for a,b,c,_ in mse_data]
values = [d for *_,d in mse_data]

plt.figure(figsize=(14,5))
bars = plt.bar(range(len(values)), values)
plt.xticks(range(len(values)), labels, rotation=60, ha='right')
plt.ylabel("MSE")
plt.title("Grafik Batang Perbandingan MSE")
plt.grid(axis='y')

for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height(),
             f"{bar.get_height():.1f}",
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()