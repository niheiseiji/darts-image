# %%

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/shomen.png"  # 読み込むPNG画像のパス
# image_path = "./img/naname.png"  # 読み込むPNG画像のパス
# image_path = "./img/toonaname.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# 画像をBGRからHSVに変換
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# 元画像の色を抽出して背景画像を緑に設定する
def extract_color_range(image_hsv, lower_black, upper_black, background_hsv=[0, 0, 0]):
    # 指定色を抽出
    mask = cv2.inRange(image_hsv, lower_black, upper_black)

    # 元画像の指定色を抽出
    result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    # 背景色を設定
    color_background = np.full_like(image_hsv, background_hsv, dtype=np.uint8)
    result_with_bg = np.where(result == 0, color_background, result)

    return result_with_bg

# 色の範囲を指定して色ごとに抽出（HSV値で指定）
# 色相は0~180
# 彩度明度は0~255

# 赤の範囲（HSV空間）
lower_red1 = np.array([0, 80, 122])
upper_red1 = np.array([25, 255, 255])
lower_red2 = np.array([160, 70, 100])
upper_red2 = np.array([180, 255, 255])

# 緑の範囲（HSV空間）
lower_green = np.array([90, 160, 80])
upper_green = np.array([100, 255, 255])

# 黒の範囲（HSV空間）
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 80])

# ベージュの範囲（HSV空間）
lower_beige = np.array([10, 15, 100])
upper_beige = np.array([35, 150, 255])

# 色ごとに抽出（赤は2つの範囲があるので結合）
red_mask1 = extract_color_range(image_hsv, lower_red1, upper_red1)
red_mask2 = extract_color_range(image_hsv, lower_red2, upper_red2)
red_extracted = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)
green_extracted = extract_color_range(image_hsv, lower_green, upper_green)
black_extracted = extract_color_range(
    image_hsv, lower_black, upper_black, [50, 255, 255]
)
beige_extracted = extract_color_range(image_hsv, lower_beige, upper_beige)

# 抽出結果を表示
plt.figure(figsize=(12, 8))

# 赤色抽出結果
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(red_extracted, cv2.COLOR_HSV2RGB))
plt.title("Red Segments")
plt.axis("off")

# 緑色抽出結果
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(green_extracted, cv2.COLOR_HSV2RGB))
plt.title("Green Segments")
plt.axis("off")

# 黒色抽出結果
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(black_extracted, cv2.COLOR_HSV2RGB))
plt.title("Black Segments")
plt.axis("off")

# ベージュ色抽出結果
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(beige_extracted, cv2.COLOR_HSV2RGB))
plt.title("Beige Segments")
plt.axis("off")

plt.show()
# %%
