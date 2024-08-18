# %%
import cv2
import numpy as np


def is_gray_region(image, contour):
    """
    四角形の内部がグレーで塗りつぶされているかを確認します。
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask)[:3]
    # グレー判定のための閾値を調整
    return np.all(np.abs(np.array(mean_val) - np.array([128, 128, 128])) < 50)


def detect_gray_rectangles(image_path):
    """
    画像内のすべての四角形を検出し、内部がグレーで塗りつぶされているものの数を表示します。
    """
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)

    # 輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 四角形の輪郭を見つける
    rect_contours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect_contours.append(approx)

    # グレーの四角形をカウント
    gray_rects_count = sum(
        1 for contour in rect_contours if is_gray_region(image, contour)
    )

    # グレーの四角形の数を表示
    print(f"検出されたグレーの四角形の数: {gray_rects_count}")

    # グレーの四角形を描画
    image_with_rects = image.copy()
    for contour in rect_contours:
        if is_gray_region(image, contour):
            cv2.drawContours(image_with_rects, [contour], -1, (0, 255, 0), 2)

    # 検出結果を表示
    plt.imshow(cv2.cvtColor(image_with_rects, cv2.COLOR_BGR2RGB))
    plt.title("Detected Gray Rectangles")
    plt.show()


# 画像ファイルのパスを指定して関数を実行
detect_gray_rectangles("./img/separate.jpg")

# %%
