# %%
import cv2

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
# %%