# %%

# ダーツ刺さる前後の差分検出でダーツ矢を抽出する
# ライティングがかなり影響する。特にカメラ方向の証明が不十分だとダーツ先端の検出がうまくできない
import cv2
import numpy as np

def detect_dart_diff(image1, image2):
    """
    2つの画像の差分を検出し、2値化した差分画像を返す関数
    :param image1: 比較対象の1枚目の画像 (numpy array)
    :param image2: 比較対象の2枚目の画像 (numpy array)
    :return: 差分画像 (numpy array)
    """

    # グレースケールに変換
    before_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

   # CLAHEによるコントラスト補正
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    before_clahe = clahe.apply(before_gray)
    after_clahe = clahe.apply(after_gray)

    # GaussianBlurで前後画像をぼかし、ノイズを軽減
    before_blurred = cv2.GaussianBlur(before_clahe, (15, 15), 0)
    after_blurred = cv2.GaussianBlur(after_clahe, (15, 15), 0)

    # 差分画像を計算
    diff_image = cv2.absdiff(before_blurred, after_blurred)

    # 差分画像のしきい値処理（しきい値を調整）
    _, thresh = cv2.threshold(diff_image, 40, 255, cv2.THRESH_BINARY)

    # モルフォロジー演算でノイズを除去
    kernel = np.ones((5, 5), np.uint8)
    cleaned_diff = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned_diff

# %%
