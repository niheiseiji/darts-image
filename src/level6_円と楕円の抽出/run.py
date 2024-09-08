# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
# image_path = "./img/shomen.png"  # 読み込むPNG画像のパス
image_path = "./img/toonaname.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# 画像をBGRからHSVに変換
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤の範囲（HSV空間）
lower_red1 = np.array([0, 80, 122])
upper_red1 = np.array([25, 255, 255])
lower_red2 = np.array([160, 70, 100])
upper_red2 = np.array([180, 255, 255])

# 赤色の抽出（赤は2つの範囲があるので結合）
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
red_extracted = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

# 輪郭の検出（マスク画像のまま使用）
contours, _ = cv2.findContours(red_extracted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 楕円を描画するために、グレースケール画像をカラーに変換
output_image = cv2.cvtColor(red_extracted, cv2.COLOR_GRAY2BGR)

# 輪郭をフィルタリングして楕円を描画
min_area = 2000000  # 調整可能: 小さな輪郭を無視するための最小面積
max_area = 3000000  # 調整可能: 大きな輪郭を無視するための最大面積
for contour in contours:
    if len(contour) >= 5 and cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < max_area:
        print(cv2.contourArea(contour))
        # 楕円をフィット
        ellipse = cv2.fitEllipse(contour)
        # 楕円を緑色で描画
        cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)

# 結果の保存
output_path = "./img/output/detected_image.png"
cv2.imwrite(output_path, output_image)

# 結果を表示
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Ellipse Fitting on Extracted Red Segments")
plt.axis("off")
plt.show()

# %%
