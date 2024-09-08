# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像の読み込み
# image_path = "./img/marktuki.png"  # 読み込むPNG画像のパス
image_path = "./img/init.png"  # 読み込むPNG画像のパス
image = cv2.imread(image_path)

# 画像をBGRからHSVに変換
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 元画像の色を抽出して背景画像を緑に設定する
def extract_color_range(image_hsv, lower_color, upper_color, background_hsv=[0, 0, 0]):
    # 指定色を抽出
    mask = cv2.inRange(image_hsv, lower_color, upper_color)

    # 元画像の指定色を抽出
    result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    # 背景色を設定
    color_background = np.full_like(image_hsv, background_hsv, dtype=np.uint8)
    result_with_bg = np.where(result == 0, color_background, result)

    return result_with_bg

# 青の範囲（HSV空間）
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([125, 255, 255])

# 青を抽出
blue_extracted = extract_color_range(image_hsv, lower_blue, upper_blue)

# 抽出した青要素をグレースケールに変換
blue_gray = cv2.cvtColor(cv2.cvtColor(blue_extracted, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

# 青要素の輪郭を検出
contours, _ = cv2.findContours(blue_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 検出されたマークの座標を保存するリスト
blue_mark_centers = []

# 各輪郭に対して処理
for contour in contours:
    # 面積が小さすぎるノイズを除去
    area = cv2.contourArea(contour)
    if area > 100:  # 調整可能な閾値
        # 輪郭の重心を計算
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            blue_mark_centers.append((cX, cY))

            # 重心を赤い円で描画
            cv2.circle(image, (cX, cY), 10, (0, 255, 0), -1)

# 上下左右の順番に並べ替えるために、重心座標を並べ替える
# まずY座標でソートして、上2つと下2つを分ける
blue_mark_centers = sorted(blue_mark_centers, key=lambda p: p[1])  # Y座標でソート

# 上下に分けた後、上の2つはX座標でソート（左→右）、下の2つも同様
top_marks = sorted(blue_mark_centers[:2], key=lambda p: p[0])  # 上側2点（X座標で左→右に並べる）
bottom_marks = sorted(blue_mark_centers[2:], key=lambda p: p[0], reverse=True)  # 下側2点（X座標で右→左に並べる）

# 上下左右の順番にした座標リスト
sorted_marks = [top_marks[0], top_marks[1], bottom_marks[0], bottom_marks[1]]

# 結果の保存
output_path = "./img/output/mark_image2.png"
cv2.imwrite(output_path, image)

# 結果の表示
plt.figure(figsize=(12, 12))
plt.subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Blue Marks with Centers (Ordered)")
plt.axis("off")
plt.show()

# 検出された各青のマークの座標（上、右、下、左の順）を出力
for i, center in enumerate(sorted_marks):
    print(f"Mark {i+1} (Up-Right-Down-Left order): X = {center[0]}, Y = {center[1]}")

# %%
