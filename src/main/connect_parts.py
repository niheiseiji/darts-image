# %%
import cv2
import numpy as np

def connect_parts(image, kernel_size=(5, 5), iterations=2):
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

# %%