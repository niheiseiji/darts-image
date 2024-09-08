# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

from mark_extractor import extract_blue_marks  # ここで別ファイルの関数をインポート
from diff_detector import detect_dart_diff  # 差分検出用
from connect_parts import connect_parts # 膨張処理
from detect_contour_bottom import detect_contour_bottom

'''
1.セットアップ処理
4点のマークから射影変換行列を取得する
'''
# 画像の読み込み
image_path = "./img/input/init/marktuki.png"  # 読み込むPNG画像のパス
init_image = cv2.imread(image_path)

# 青いマークの座標を取得（上右下左の順)
sorted_marks = extract_blue_marks(image_path)  # 別ファイルの関数を使って座標を取得
# 変換後の座標（上右下左に対応）
output_points = np.array([[250, 0], [500, 250], [250, 500], [0, 250]], dtype="float32")

# 射影変換行列を取得
matrix = cv2.getPerspectiveTransform(np.array(sorted_marks, dtype="float32"), output_points)

'''
1の検証コード
'''
# 射影変換を適用して、500x500の画像に変換
output_size = (500, 500)
warped_image = cv2.warpPerspective(init_image, matrix, output_size)

# 結果の保存
output_path = "./img/output/warped_image.png"
cv2.imwrite(output_path, warped_image)

# 結果の表示
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Warped Image (500x500)")
plt.axis("off")
plt.show()
'''
1の検証コード 終わり
'''

'''
2.ダーツ検出
前後画像の差分からダーツオブジェクトを検出する
'''
image_path = "./img/input/darts/bull_point.png"
dart_image = cv2.imread(image_path)

diff_image = detect_dart_diff(init_image, dart_image)

'''
2の検証コード
'''
# 差分画像を保存
output_path = "./img/output/diff_image.png"
cv2.imwrite(output_path, diff_image)

# 差分画像の表示
plt.figure(figsize=(6, 6))
plt.imshow(diff_image, cmap='gray')
plt.title("Difference (Binary)")
plt.axis("off")
plt.show()
'''
2の検証コード 終わり
'''

'''
3.ダーツが刺さった座標を特定する
差分画像のうちダーツオブジェクトを検出して刺さった座標を検出する
'''
# 途切れた物体を接続する
connected_image = connect_parts(diff_image, kernel_size=(20, 20), iterations=2)

# 一定以上の大きさのオブジェクトを検出し、その最下部の座標を取得
min_area = 5000  # 最小面積の閾値を設定
detected_image, bottom_coords = detect_contour_bottom(connected_image, min_area)

'''
3の検証コード
'''
# 差分画像を保存
output_path = "./img/output/connected_image.png"
cv2.imwrite(output_path, connected_image)

# 差分画像の表示
plt.figure(figsize=(6, 6))
plt.imshow(connected_image, cmap='gray')
plt.title("Connected (Binary)")
plt.axis("off")
plt.show()

# ダーツオブジェクト検出画像を保存
output_path = "./img/output/detected_dart_image.png"
cv2.imwrite(output_path, detected_image)
# ダーツオブジェクト検出画像の表示
plt.figure(figsize=(6, 6))
plt.imshow(detected_image, cmap='gray')
plt.title("Dart (Binary)")
plt.axis("off")
plt.show()
'''
3の検証コード 終わり
'''

'''
4.ダーツの刺さった座標を射影変換して青い点を描画する
'''
# bottom_coordsを射影変換
for i, bottom in enumerate(bottom_coords):
    # 座標をリスト形式にしてから射影変換を適用
    src_point = np.array([[bottom]], dtype='float32')  # 1x2の座標を射影変換
    dst_point = cv2.perspectiveTransform(src_point, matrix)

    # 変換後の座標を整数に変換
    x, y = int(dst_point[0][0][0]), int(dst_point[0][0][1])

    # 青い点を描画
    cv2.circle(warped_image, (x, y), 4, (0, 255, 0), -1)

# 青い点が描画された結果の保存と表示
output_path = "./img/output/warped_image_with_dart.png"
cv2.imwrite(output_path, warped_image)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.ylabel("y")
plt.xlabel("x")
plt.title("Warped Image with Dart Locations")
plt.axis("off")
plt.show()
'''
4の検証コード 終わり
'''

# %%
