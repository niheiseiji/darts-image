# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt


def connect_broken_parts(image, kernel_size=(7, 7), iterations=2):
    """
    途切れた物体を一つの物体として扱うために膨張と収縮を行う処理
    :param image: 二値化された入力画像
    :param kernel_size: モルフォロジー演算で使用するカーネルサイズ
    :param iterations: 膨張処理の回数
    :return: 処理された画像
    """
    # カーネルの作成
    kernel = np.ones(kernel_size, np.uint8)

    # 膨張処理で物体を接続
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)

    # 収縮処理で形を戻す（膨張しすぎた部分を縮小）
    connected_image = cv2.erode(dilated_image, kernel, iterations=iterations)

    return connected_image


def detect_contour_bottom(image, min_area=1000):
    """
    一定以上の面積を持つオブジェクトを検出し、その最下部の座標を取得
    :param image: 二値化された入力画像
    :param min_area: 最小面積の閾値
    :return: 検出されたオブジェクトを描画した画像、物体の最下部の座標リスト
    """
    # 輪郭を検出
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # オブジェクトの輪郭を描画するための出力画像
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 検出された物体の最下部の座標リスト
    bottom_points = []

    # 一定以上の面積を持つオブジェクトをフィルタリングして描画
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # 最下部のY座標を持つ点を取得
            bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
            bottom_points.append(bottom_point)

            # 輪郭を描画
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

            # 最下部の点を描画（赤い円）
            cv2.circle(output_image, bottom_point, 20, (0, 0, 255), -1)

    return output_image, bottom_points


# 画像の読み込み（既に処理された二値画像を使用）
image_path = "./img/cleaned_diff.png"  # 既に二値化された画像を想定
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 途切れた物体を接続する処理を実行
connected_image = connect_broken_parts(image, kernel_size=(15, 15), iterations=2)

# 一定以上の大きさのオブジェクトを検出し、その最下部の座標を取得
min_area = 5000  # 最小面積の閾値を設定
detected_image, bottom_coords = detect_contour_bottom(connected_image, min_area)

# 検出された物体の最下部の座標を表示
for i, bottom in enumerate(bottom_coords):
    print(f"Object {i+1} Bottom Point: {bottom}")

# 結果の保存
output_path = "./img/output/detected_objects_with_bottom.png"
cv2.imwrite(output_path, detected_image)

# 結果の表示
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image (Broken)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(connected_image, cmap="gray")
plt.title("Connected Image (Merged Object)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Objects with Bottom Points")
plt.axis('off')

plt.show()
# %%
