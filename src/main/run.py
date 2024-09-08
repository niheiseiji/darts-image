# %%
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from get_score import get_score
from mark_extractor import extract_blue_marks  # ここで別ファイルの関数をインポート
from diff_detector import detect_dart_diff  # 差分検出用
from connect_parts import connect_parts  # 膨張処理
from detect_contour_bottom import detect_contour_bottom

"""
0.入力画像パスの設定
"""
INIT_PATH = "./img/input/init/init.png"
DART_PATH = "./img/input/darts/t13.png"
OUTPUT_DIR = "./img/output/t13/"

"""
1.セットアップ処理
4点のマークから射影変換行列を取得する
"""
# 画像の読み込み
image_path = INIT_PATH
init_image = cv2.imread(image_path)

# 青いマークの座標を取得（上右下左の順)※blue_extracted_imageはデバッグ用
sorted_marks, blue_extracted_image = extract_blue_marks(image_path)

# 変換後の座標（上右下左の順）
output_points = np.array([[250, 0], [500, 250], [250, 500], [0, 250]], dtype="float32")

# 射影変換行列を取得
matrix = cv2.getPerspectiveTransform(
    np.array(sorted_marks, dtype="float32"), output_points
)

# 射影変換を適用して、500x500の画像に変換
output_size = (500, 500)
warped_image = cv2.warpPerspective(init_image, matrix, output_size)

# warped_imageのコピーを作成して、直接更新しないようにする
warped_image_copy = warped_image.copy()

"""
2.ダーツ検出
前後画像の差分からダーツオブジェクトを検出する
"""
image_path = DART_PATH
dart_image = cv2.imread(image_path)

diff_image = detect_dart_diff(init_image, dart_image)

"""
3.ダーツが刺さった座標を特定する
差分画像のうちダーツオブジェクトを検出して刺さった座標を検出する
"""
# 途切れた物体を接続する
connected_image = connect_parts(diff_image, kernel_size=(24, 24), iterations=2)

# 一定以上の大きさのオブジェクトを検出し、その最下部の座標を取得
min_area = 10000  # 最小面積の閾値を設定
detected_image, bottom_coords = detect_contour_bottom(connected_image, min_area)

"""
4.ダーツの刺さった座標を射影変換して青い点を描画する
"""
# bottom_coordsを射影変換
for i, bottom in enumerate(bottom_coords):
    # 座標をリスト形式にしてから射影変換を適用
    src_point = np.array([[bottom]], dtype="float32")  # 1x2の座標を射影変換
    dst_point = cv2.perspectiveTransform(src_point, matrix)

    # 変換後の座標を整数に変換
    x, y = int(dst_point[0][0][0]), int(dst_point[0][0][1])

    # 青い点を描画
    cv2.circle(warped_image_copy, (x, y), 4, (0, 255, 0), -1)

"""
5.(x,y)から得点計算
"""
# 座標 (x, y) を入力して得点を計算
score = get_score(x, y)
print(f"Dart at ({x}, {y}) scores: {score} points")
# %%

# 以下検証コード
"""
1の検証コード
"""
plt.figure(figsize=(12, 6))
# 初期画像の表示（BGRからRGBに変換して表示）
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))  # BGRからRGBへ変換
plt.title("initial image")
plt.axis("off")
# 青抽出結果の表示
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(cv2.cvtColor(blue_extracted_image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))
plt.title("Blue Extracted")
# 射影変換の確認
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # BGRからRGBへ変換
plt.title("projective transformation (500x500)")
plt.axis("off")
"""
2の検証コード
"""
plt.figure(figsize=(12, 6))
# 初期画像の表示（BGRからRGBに変換して表示）
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))  # BGRからRGBへ変換
plt.title("initial image")
plt.axis("off")
# ダーツ画像の表示（BGRからRGBに変換して表示）
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(dart_image, cv2.COLOR_BGR2RGB))  # BGRからRGBへ変換
plt.title("darts image")
plt.axis("off")
# 差分画像の表示（グレースケールなのでcmap='gray'を使用）
plt.subplot(1, 3, 3)
plt.imshow(diff_image, cmap="gray")
plt.title("diff image (Binary)")
plt.axis("off")
"""
3の検証コード
"""
plt.figure(figsize=(12, 6))
# 差分画像の表示（グレースケールなのでcmap='gray'を使用）
plt.subplot(1, 2, 1)
plt.imshow(connected_image, cmap="gray")
plt.title("connected image (Binary)")
plt.axis("off")
# ダーツオブジェクト検出画像の表示（グレースケール）
plt.subplot(1, 2, 2)
plt.imshow(detected_image, cmap="gray")
plt.title("detected darts point (Binary)")
plt.axis("off")
# 4検証
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(warped_image_copy, cv2.COLOR_BGR2RGB))  # BGRからRGBへ変換
# X軸とY軸にメモリを設定
plt.xticks(np.arange(0, warped_image.shape[1], step=50))
plt.yticks(np.arange(0, warped_image.shape[0], step=50))
plt.title("mapping darts point")
plt.axis("on")

plt.show()

# 座標 (x, y) を入力して得点を計算
score = get_score(x, y)
print(f"Dart at ({x}, {y}) scores: {score} points")

# %%
# 以下保存系のコード

# ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4点抽出結果を保存
output_path = os.path.join(OUTPUT_DIR, "detect_blue_point.png")
cv2.imwrite(output_path, blue_extracted_image)
# 射影変換結果を保存
output_path = os.path.join(OUTPUT_DIR, "projective_transformation.png")
cv2.imwrite(output_path, warped_image)
# 差分画像を保存
output_path = os.path.join(OUTPUT_DIR, "diff_image.png")
cv2.imwrite(output_path, diff_image)
# 膨張画像を保存
output_path = os.path.join(OUTPUT_DIR, "connected_image.png")
cv2.imwrite(output_path, connected_image)
# ダーツオブジェクト検出画像を保存
output_path = os.path.join(OUTPUT_DIR, "detected_dart_image.png")
cv2.imwrite(output_path, detected_image)
# 青い点が描画された結果の保存
output_path = os.path.join(OUTPUT_DIR, "warped_image_with_dart.png")
cv2.imwrite(output_path, warped_image_copy)

# %%
