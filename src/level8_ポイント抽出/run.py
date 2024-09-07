# %%

# 画像内のポイントの色だけを抽出したい

import cv2
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/up.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# GaussianBlurでノイズを除去（エッジ検出前の前処理）
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Cannyエッジ検出を適用
# 参考値
# 高コントラスト画像（明確な境界がある画像）:
# threshold1 = 50
# threshold2 = 150 ～ 200
# 低コントラスト画像（境界が曖昧な画像）:
# threshold1 = 30
# threshold2 = 100 ～ 150
# ノイズの多い画像:
# threshold1 = 100
# threshold2 = 200 ～ 250
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# 結果を表示
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis("on")

plt.show()

# %%
