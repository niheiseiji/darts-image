# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

from util.jpg2png import convert_jpg_to_png

# 画像を読み込む
# image_path = convert_jpg_to_png("./img/board_light.jpg")
image_path = "./img/board_light.png"
image = cv2.imread(image_path)

# ガウシアンブラーで表面を滑らかに
gaussianblur_image = cv2.GaussianBlur(image, (51, 51), 0)
# 処理後の画像を保存(確認用)
output_path = "./img/gaussianblur_image.png"
cv2.imwrite(output_path, gaussianblur_image)

# 結果を表示
# plt.subplot(1, 2, 1)
# plt.title('GaussianBlur Image')
# plt.imshow(cv2.cvtColor(gaussianblur_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # 軸を非表示

# 色の範囲を指定してマスクを作成し、色を抽出する関数
def extract_color_range(image, lower_bound, upper_bound, bg_color=(0, 0, 0)):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 背景を指定された色に設定する
    background = np.full_like(image, bg_color, dtype=np.uint8)
    result_with_bg = np.where(result == 0, background, result)

    return result_with_bg

# 赤をBGRで抽出する
lower_red = np.array([0, 0, 100])
upper_red = np.array([80, 80, 255])
red_extracted = extract_color_range(gaussianblur_image, lower_red, upper_red)

# 緑をBGRで抽出する
lower_green = np.array([0, 100, 0])
upper_green = np.array([115, 177, 118])
green_extracted = extract_color_range(gaussianblur_image, lower_green, upper_green)

# ベージュをBGRで抽出する
lower_beige = np.array([41, 80, 133])
upper_beige = np.array([117, 187, 239])
beige_extracted = extract_color_range(gaussianblur_image, lower_beige, upper_beige)

# 黒をBGRで抽出する(精度悪いかつ調整不可)
lower_black = np.array([0, 0, 0])
upper_black = np.array([92, 101,107])
black_extracted = extract_color_range(gaussianblur_image, lower_black, upper_black, bg_color=(255, 255, 255))  # 白背景

# 結果を表示
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(red_extracted, cv2.COLOR_BGR2RGB))
plt.title("Red Segments")
plt.axis("off")  # 軸を非表示

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(green_extracted, cv2.COLOR_BGR2RGB))
plt.title("Green Segments")
plt.axis("off")  # 軸を非表示

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(beige_extracted, cv2.COLOR_BGR2RGB))
plt.title("Beige Segments")
plt.axis("off")  # 軸を非表示

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(black_extracted, cv2.COLOR_BGR2RGB))
plt.title("Black Segments")
plt.axis("off")  # 軸を非表示

plt.show()
# %%
