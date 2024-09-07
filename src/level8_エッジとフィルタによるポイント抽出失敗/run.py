# %%

# 画像内のポイントの位置を抽出したい
# エッジ検出をフィルタ処理でダーツの先端をとらえようとしたけど、
# 先端の形状と色の両方の特徴がなくて無理そう。次のレベルではパターン検出を試す

import cv2
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/up.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# GaussianBlurでノイズを除去（エッジ検出前の前処理）
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title("blurred Image")
plt.axis("off")

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
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.title("edges Image")
plt.axis("off")

# 輪郭を検出
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ダーツの先端となる細長い形状をフィルタリングする
min_aspect_ratio = 1.0  # アスペクト比の最小値 (幅に対する高さの比)
min_area = 100  # 最小の面積（小さすぎるノイズを除外するため）

output_image = image.copy()

for contour in contours:
    # 輪郭を囲む最小の外接矩形を計算
    x, y, w, h = cv2.boundingRect(contour)

    # アスペクト比（高さに対する幅の比率）を計算
    aspect_ratio = float(h) / float(w)

    # 面積を計算
    area = cv2.contourArea(contour)

    # 細長い形状かつ、一定の大きさを持つものをフィルタリング
    if aspect_ratio > min_aspect_ratio and area > min_area:
    # フィルタリングされた輪郭を描画（緑色で枠を描画）
       cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(12, 12))
# オリジナル画像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 細長い形状をフィルタリングした結果
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Filtered Thin Shapes (Darts Tip)")
plt.axis("off")

plt.show()

# %%