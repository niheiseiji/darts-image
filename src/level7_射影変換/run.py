# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/zahyo.png"
image = cv2.imread(image_path)

# 画像をグレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 緑色の4点の座標を手動で設定(ここを自動化できれば射影変換行列を自動で求められる)
src_points = np.array([
    [177, 168],  # 上
    [388, 222],  # 右
    [354, 436],  # 下
    [104, 370],  # 左
], dtype="float32")

# 正面から見た円の4つの点に対応
dst_points = np.array([
    [250, 150],  # 上
    [400, 300],  # 右
    [250, 450],  # 下
    [100, 300],  # 左
], dtype="float32")

# 射影変換行列を計算
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 射影変換を適用
warped_image = cv2.warpPerspective(image, M, (600, 600))

# 結果を表示
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Transformed to Circle")
plt.axis("off")

plt.show()

# %%
