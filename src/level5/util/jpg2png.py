import cv2
import os

def convert_jpg_to_png(jpg_image_path, output_directory=None):
    """
    JPEG画像を読み込み、PNG形式で保存する関数。

    Parameters:
    - jpg_image_path: str
        入力のJPEG画像のパス。
    - output_directory: str, optional
        出力するPNG画像を保存するディレクトリ。指定しない場合は、JPEG画像と同じディレクトリに保存されます。

    Returns:
    - png_image_path: str
        保存されたPNG画像のパス。
    """
    # JPEG画像を読み込む
    image = cv2.imread(jpg_image_path)

    # 出力ディレクトリを指定しない場合は、入力画像と同じディレクトリに保存する
    if output_directory is None:
        output_directory = os.path.dirname(jpg_image_path)

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_directory, exist_ok=True)

    # 出力ファイル名の設定
    base_name = os.path.splitext(os.path.basename(jpg_image_path))[0]
    png_image_path = os.path.join(output_directory, base_name + ".png")

    # PNG形式で保存する
    cv2.imwrite(png_image_path, image)

    print(f"Image saved as PNG: {png_image_path}")
    return png_image_path
