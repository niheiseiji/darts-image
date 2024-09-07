# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/shomen.png"  # 読み込むPNG画像のパス
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

# 楕円を円に変換するための4点を取得
min_area = 3000000  # 調整可能: 小さな輪郭を無視するための最小面積
max_area = 4000000  # 調整可能: 大きな輪郭を無視するための最大面積
for contour in contours:
    if len(contour) >= 5 and cv2.contourArea(contour) > min_area and cv2.contourArea(contour) < max_area:
        # 楕円をフィット
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse

        # 楕円の長軸・短軸の半径（長さの半分）
        major_axis = axes[1] / 2
        minor_axis = axes[0] / 2
        angle_rad = np.deg2rad(angle)  # 楕円の回転角度をラジアンに変換

        # 楕円の4つの端点（上・下・左・右）
        top_point = (
            int(center[0] - major_axis * np.sin(angle_rad)),
            int(center[1] + major_axis * np.cos(angle_rad))
        )
        bottom_point = (
            int(center[0] + major_axis * np.sin(angle_rad)),
            int(center[1] - major_axis * np.cos(angle_rad))
        )
        left_point = (
            int(center[0] - minor_axis * np.cos(angle_rad)),
            int(center[1] - minor_axis * np.sin(angle_rad))
        )
        right_point = (
            int(center[0] + minor_axis * np.cos(angle_rad)),
            int(center[1] + minor_axis * np.sin(angle_rad))
        )

        # 楕円の中心を白色で描画
        cv2.circle(output_image, (int(center[0]), int(center[1])), 100, (0, 255, 0), -1)  # 白色

        # 4つの点を画像に描画（異なる色で描画）
        cv2.circle(output_image, top_point, 100, (0, 255, 0), -1)    # 緑色
        cv2.circle(output_image, bottom_point, 100, (255, 0, 0), -1)  # 青色
        cv2.circle(output_image, left_point, 100, (0, 0, 255), -1)    # 赤色
        cv2.circle(output_image, right_point, 100, (255, 255, 0), -1) # 黄色

        # 楕円を描画
        cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)

# 結果を表示
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Ellipse with 4 Main Points")
plt.axis("off")
plt.show()


# %%
