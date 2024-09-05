# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
image_path = "./img/shomen.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# 画像をBGRからHSVに変換
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 元画像の色を抽出して背景画像を設定する関数
def extract_color_range(image_hsv, lower_bound, upper_bound, background_hsv=[0, 0, 0]):
    # 指定色を抽出
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # 元画像の指定色を抽出
    result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    # 背景色を設定
    color_background = np.full_like(image_hsv, background_hsv, dtype=np.uint8)
    result_with_bg = np.where(result == 0, color_background, result)

    return result_with_bg

# 赤の範囲（HSV空間）
lower_red1 = np.array([0, 80, 122])
upper_red1 = np.array([25, 255, 255])
lower_red2 = np.array([160, 70, 100])
upper_red2 = np.array([180, 255, 255])

# 赤色の抽出（赤は2つの範囲があるので結合）
red_mask1 = extract_color_range(image_hsv, lower_red1, upper_red1)
red_mask2 = extract_color_range(image_hsv, lower_red2, upper_red2)
red_extracted = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)

# 赤色抽出結果をグレースケールに変換
red_extracted_gray = cv2.cvtColor(cv2.cvtColor(red_extracted, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

# --- ハフ変換で円検出 ---
# グレースケール画像でガウシアンブラーを適用
blurred = cv2.GaussianBlur(red_extracted_gray, (9, 9), 2)

# グレースケール画像をカラーに変換（円の描画用）
blurred_color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# ハフ変換で円を検出
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,  # 検出精度のスケール (解像度比率)
    minDist=1000,  # 円の中心間の最小距離
    param1=100,  # Cannyエッジ検出用の上限閾値
    param2=30,  # 円検出の閾値（大きいほど厳密）
    minRadius=700,  # 検出する円の最小半径
    maxRadius=1500,  # 検出する円の最大半径
)

# 円検出された場合、カラー化されたグレースケール画像に円を描画
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for x, y, r in circles:
        # 円を描画（緑色）
        cv2.circle(blurred_color, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(
            blurred_color, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), -1
        )  # 中心点を描画

# 結果を表示
plt.figure(figsize=(20, 15))

# グレースケール画像に円検出結果を表示
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(blurred_color, cv2.COLOR_BGR2RGB))
plt.title("Grayscale with Circles")
plt.axis("off")

plt.show()


# %%
