import cv2
import numpy as np


def extract_blue_marks(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 青の範囲（HSV空間）
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([125, 255, 255])

    # 青を抽出
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    blue_extracted = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    # 抽出した青要素をグレースケールに変換
    blue_gray = cv2.cvtColor(
        cv2.cvtColor(blue_extracted, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY
    )

    # 青要素の輪郭を検出
    contours, _ = cv2.findContours(
        blue_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

    # 上下左右の順番に並べ替えるために、重心座標を並べ替える
    # まずY座標でソートして、上2つと下2つを分ける
    blue_mark_centers = sorted(blue_mark_centers, key=lambda p: p[1])  # Y座標でソート

    # 上下に分けた後、上の2つはX座標でソート（左→右）、下の2つも同様
    top_marks = sorted(
        blue_mark_centers[:2], key=lambda p: p[0]
    )  # 上側2点（X座標で左→右に並べる）
    bottom_marks = sorted(
        blue_mark_centers[2:], key=lambda p: p[0], reverse=True
    )  # 下側2点（X座標で右→左に並べる）

    # 上右下左の順にした座標リストを返す
    return [top_marks[0], top_marks[1], bottom_marks[0], bottom_marks[1]]
