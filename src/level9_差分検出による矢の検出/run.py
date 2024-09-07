# %%

# ダーツ刺さる前後の差分検出でダーツ矢を抽出する
# ライティングがかなり影響する。特にカメラ方向の証明が不十分だとダーツ先端の検出がうまくできない

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 前後画像の読み込み
before_image_path = "./img/before3.png"  # ダーツが刺さる前の画像
after_image_path = "./img/after3.png"    # ダーツが刺さった後の画像

before_image = cv2.imread(before_image_path)
after_image = cv2.imread(after_image_path)

# グレースケールに変換
before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

# GaussianBlurで前後画像をぼかし、ノイズを軽減
before_blurred = cv2.GaussianBlur(before_gray, (9, 9), 0)  # カーネルサイズを少し小さく
after_blurred = cv2.GaussianBlur(after_gray, (9, 9), 0)

# 差分画像を計算
diff_image = cv2.absdiff(before_blurred, after_blurred)

# 差分画像のしきい値処理（しきい値を少し下げて微妙な変化も捉える）
_, thresh = cv2.threshold(diff_image, 40, 255, cv2.THRESH_BINARY)

# モルフォロジー演算（カーネルサイズを少し小さく）
kernel = np.ones((5, 5), np.uint8)
cleaned_diff = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 結果を表示
plt.figure(figsize=(12, 8))

# オリジナルの前画像
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB))
plt.title("Before Image")
plt.axis("off")

# オリジナルの後画像
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB))
plt.title("After Image")
plt.axis("off")

# 差分画像（ノイズ除去後）の表示
plt.subplot(1, 3, 3)
plt.imshow(cleaned_diff, cmap='gray')
plt.title("Difference (Adjusted for Dart Tip)")
plt.axis("off")

plt.show()

# %%
