# %%
# ブル円を検出してその中にダーツが入っているか検出するアプローチ
# →失敗。丸の検出でブルを検出できない。画像内の色んなマルがあるしブルに刺さっているとブルが画像から見えずらい
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 画像を読み込む
image_path = "./img/dart_bull_one.jpg"
image = cv2.imread(image_path)

# グレースケールに変換
## 計算負荷・リソース負荷の軽減。各ピクセルが明度値の違いしかなくなる。色情報が不要な場合は使う。
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# グレスケ画像を表示する
# plt.figure(figsize=(6, 6))  # 新たな図の作成6cm*6cm
# plt.imshow(
#     gray, cmap="gray"
# )  # 画像を表示する。カラーマップを指定する。カラーマップは数値データと色のマッピング。
# plt.title("Grayscale Image")  # 画像にタイトルつけるだけ
# plt.axis("on")  # 軸を非表示にする
# plt.show()  # 画像を表示する

# ガウシアンブラーでノイズ除去
# 画像のディティールを除去する
# blurred = cv2.GaussianBlur(gray, (5, 5), 2)
blurred = cv2.GaussianBlur(gray, (11, 11), 2)
# グレスケ画像を表示する
plt.figure(figsize=(6, 6))  # 新たな図の作成6cm*6cm
plt.imshow(
    blurred, cmap="gray"
)  # 画像を表示する。カラーマップを指定する。カラーマップは数値データと色のマッピング。
plt.title("Blurred Image")  # 画像にタイトルつけるだけ
plt.axis("on")  # 軸を非表示にする
plt.show()  # 画像を表示する

# Cannyエッジ検出を適用
# edges = cv2.Canny(blurred, 1, 150)
# edges = cv2.Canny(blurred, 50, 150)# ノイズ多すぎ
# edges = cv2.Canny(blurred, 100, 150)# ノイズ多すぎ
edges = cv2.Canny(blurred, 100, 200)
# edges = cv2.Canny(blurred, 200, 300)# 暗すぎ
plt.figure(figsize=(6, 6))  # 新たな図の作成6cm*6cm
plt.imshow(
    edges, cmap="gray"
)  # 画像を表示する。カラーマップを指定する。カラーマップは数値データと色のマッピング。
plt.title("edges Image")  # 画像にタイトルつけるだけ
plt.axis("on")  # 軸を非表示にする
plt.show()  # 画像を表示する

# ハフ変換による円検出
circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=40, # 大きいほど厳密な円検出
    minRadius=30,
    maxRadius=50,
)

# 検出された円を画像に描画
output = image.copy()
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)  # 円を描画
        cv2.rectangle(
            output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1
        )  # 中心点を描画

# 結果を表示
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Circles")
plt.axis("off")  # 軸を非表示にする
plt.show()

# %%
