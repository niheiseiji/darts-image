# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
このスクリプトは、画像中の黒い四角の中に赤いドットがあるかどうかを判定するためのものです。
画像を読み込み、グレースケール変換、エッジ検出、輪郭抽出を行い、最大の四角形を特定します。
次に、その四角形の中に赤いドットが含まれているかどうかをHSV色空間で検出します。
"""

def detect_red_dot(image_path):
    """
    黒い四角の中に赤いドットがあるか判定します
    """
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # 画像の表示
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # グレースケール画像の表示
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.show()

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # エッジ画像の表示
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.show()

    # 輪郭の検出
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を探す
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # 四角の範囲を定義
    rect_top_left = (x, y)
    rect_bottom_right = (x + w, y + h)

    # 四角を描画して表示
    image_with_rect = image.copy()
    cv2.rectangle(image_with_rect, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB))
    plt.title("Detected Rectangle")
    plt.show()

    # HSV色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV画像の表示
    plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title("HSV Image")
    plt.show()

    # 赤色の範囲を定義
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 赤色のマスクを作成
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # マスク画像の表示
    plt.imshow(mask, cmap="gray")
    plt.title("Red Mask")
    plt.show()

    # 四角の範囲内に赤点があるかどうかを判定
    red_in_rect = False
    for y in range(rect_top_left[1], rect_bottom_right[1]):
        for x in range(rect_top_left[0], rect_bottom_right[0]):
            if mask[y, x] > 0:
                red_in_rect = True
                break
        if red_in_rect:
            break

    # 結果を表示
    if red_in_rect:
        print("true__枠内に赤点あり")
    else:
        print("false__枠内に赤点なし")

    # 四角を描画して表示
    cv2.rectangle(image, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Final Result")
    plt.show()


# 画像ファイルのパスを指定して関数を実行
detect_red_dot("./img/off.jpg")
detect_red_dot("./img/off_2.jpg")
detect_red_dot("./img/on.jpg")
detect_red_dot("./img/on_2.jpg")

# %%
