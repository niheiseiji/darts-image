# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
# image_path = "./img/zahyo.png"
image_path = "./img/zahyo2.png"
image = cv2.imread(image_path)

# 画像をグレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 緑色の4点の座標を手動で設定(ここを自動化できれば射影変換行列を自動で求められる)
# src_points = np.array([
#     [177, 168],  # 上
#     [388, 222],  # 右
#     [354, 436],  # 下
#     [104, 370],  # 左
# ], dtype="float32")
src_points = np.array([
    [1090, 587],  # 上
    [2363, 703],  # 右
    [2453, 1525],  # 下
    [379, 1257],  # 左
], dtype="float32")

# 正面から見た円の4つの点に対応
# dst_points = np.array([
#     [250, 150],  # 上
#     [400, 300],  # 右
#     [250, 450],  # 下
#     [100, 300],  # 左
# ], dtype="float32")
dst_points = np.array([
    [1343, 412],  # 上
    [2090, 1164],  # 右
    [1342, 1908],  # 下
    [590, 1162],  # 左
], dtype="float32")

# 射影変換行列を計算
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 射影変換を適用
warped_image = cv2.warpPerspective(image, M, (3200, 3200))

# 結果の保存
output_path = "./img/output/warped_image.png"
cv2.imwrite(output_path, warped_image)


# 結果を表示
plt.figure(figsize=(20, 20))
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
