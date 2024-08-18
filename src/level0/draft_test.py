# TODO: テストデータ要修正
# %%
# オリジナル、グレスケ、エッジ検出のサンプル
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image_path = "test.jpg"  # 画像ファイルのパスを指定
image = cv2.imread(image_path)

# 画像の表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# グレースケール変換
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# グレースケール画像の表示
plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# エッジ検出
edges = cv2.Canny(gray_image, 100, 200)

# エッジ画像の表示
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")

plt.show()

# %%
