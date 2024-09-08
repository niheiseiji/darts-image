# %%
import math

print("get_score")

# ダーツボードのセクター（1-20の順序）
# sector_values = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
sector_values = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13]

# ダーツボードの各エリアの半径 (mm)
bullseye_radius = 6.35  # インブル
bull_radius = 16  # ブル
inner_single_radius = 99  # シングル
triple_ring_radius = 107  # トリプル
outer_single_radius = 162  # シングル
double_ring_radius = 170  # ダブル
board_edge_radius = 225.5  # ボード端

# 画像サイズ 500x500 におけるボード中心 (250, 250)
image_center_x = 250
image_center_y = 250

# 実際のボード半径 (225.5mm) を画像の250ピクセルに対応させる
scale_factor = 250 / board_edge_radius  # 実寸(mm)からピクセルへの変換比率


def get_score(x, y):
    print("get_score")

    """
    ダーツの座標 (x, y) を基に得点を計算する関数
    :param x: 画像内のx座標
    :param y: 画像内のy座標
    :return: ダーツの得点
    """
    # 中心からの距離を計算
    dx = x - image_center_x
    dy = y - image_center_y
    distance = math.sqrt(dx**2 + dy**2) / scale_factor  # 距離をピクセルからmmに変換

    # インブルエリア
    if distance <= bullseye_radius:
        return 50  # インブルは50点
    # ブルエリア
    elif distance <= bull_radius:
        return 25  # ブルは25点
    # ダーツがボードの範囲外
    elif distance > board_edge_radius:
        return 0  # 範囲外は0点

    # 角度を計算して、セクターを判定
    angle = math.degrees(math.atan2(dy, dx))  # ラジアンから角度に変換
    if angle < 0:
        angle += 360  # 負の角度を正の角度に変換

    # セクターは18度ごとに配置されている（20セクター）
    sector_index = int(((angle + 9) % 360) // 18)
    score = sector_values[sector_index]
    print("angle", angle)
    print("sector_index", sector_index)

    # ダブルリングエリア
    if outer_single_radius < distance <= double_ring_radius:
        print("double")
        return score * 2
    # トリプルリングエリア
    elif inner_single_radius < distance <= triple_ring_radius:
        print("triple")
        return score * 3
    # シングルエリア
    print("single")
    return score


# %%
# 検証
get_score(250, 400) # 3
get_score(100, 250) # 11
get_score(250, 100) # 20
get_score(400, 250) # 6
# %%
