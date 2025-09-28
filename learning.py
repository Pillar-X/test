#
# import numpy as np
# import random
# import pandas as pd
#
# # ---------- 读点并加端点 ----------
# df = pd.read_excel('Simulated_annealing.xlsx')
# xs = df.iloc[:, 0].tolist()
# ys = df.iloc[:, 1].tolist()
#
# # 在首尾各加一个端点（示例：70,40）
# xs = [70] + xs + [70]
# ys = [40] + ys + [40]
# n = len(xs)                     # 期望 n = 原始点数 + 2
#
# # ---------- 球面距离 ----------
# def get_dis(r1, r2, R=6371000.0):
#     lon1, lat1 = np.radians(r1[0]), np.radians(r1[1])
#     lon2, lat2 = np.radians(r2[0]), np.radians(r2[1])
#     va = np.array([R*np.cos(lat1)*np.cos(lon1),
#                    R*np.cos(lat1)*np.sin(lon1),
#                    R*np.sin(lat1)])
#     vb = np.array([R*np.cos(lat2)*np.cos(lon2),
#                    R*np.cos(lat2)*np.sin(lon2),
#                    R*np.sin(lat2)])
#     cos_theta = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     return R * np.arccos(cos_theta)
#
# # ---------- 距离矩阵 ----------
# mat = np.zeros((n, n))
# for i in range(n):
#     for j in range(i+1, n):
#         r1 = (xs[i], ys[i])
#         r2 = (xs[j], ys[j])
#         d = get_dis(r1, r2)
#         mat[i, j] = mat[j, i] = d
#
# # ---------- 初始路径（开链：0 -> ... -> n-1） ----------
# inner = list(range(1, n-1))             # 只打乱内部点
# best_path = [0] + random.sample(inner, len(inner)) + [n-1]
#
# def path_len(p):
#     return sum(mat[p[k-1], p[k]] for k in range(1, len(p)))
#
# best_len = path_len(best_path)
# print("init:", best_len)
#
# # ---------- 模拟退火 + 2-opt ----------
# T = 1.0
# alpha = 0.999
# L = 20000
# eps = 1e-30
# path = best_path[:]
# plen = best_len
#
# for _ in range(L):
#     # 在 1..n-2 中选两个不同断点（开链，避免端点越界）
#     i, j = sorted(random.sample(range(1, n-1), 2))
#
#     a, b = path[i-1], path[i]
#     c, d = path[j],   path[j+1]
#
#     # 2-opt 增量（只替换边 (a,b),(c,d) -> (a,c),(b,d)）
#     df = (mat[a, c] + mat[b, d]) - (mat[a, b] + mat[c, d])
#
#     # 接受准则
#     if df < 0 or np.random.rand() < np.exp(-df / T):
#         path[i:j+1] = reversed(path[i:j+1])
#         plen += df
#         if plen < best_len:
#             best_len = plen
#             best_path = path[:]
#
#     T *= alpha
#     if T < eps:
#         break
#
# print("best:", best_len)
# # best_path 就是结果

import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

print("haha")


# -----------------------------
# 1) 读取数据：第一列经度 -> x_list；第二列纬度 -> y_list
#    文件第一行是表头，pandas 默认会当作表头处理
# -----------------------------
def load_points_from_excel(path: str) -> Tuple[List[float], List[float]]:
    df = pd.read_excel(path, header=0, usecols=[0, 1])
    # 确保恰好读取到 100 个点（不强求，但可校验）
    # assert len(df) == 100, f"期望 100 行数据，实际 {len(df)} 行"

    x_list = df.iloc[:, 0].astype(float).tolist()  # 经度
    y_list = df.iloc[:, 1].astype(float).tolist()  # 纬度
    return x_list, y_list


# -----------------------------
# 2) 在首尾各加入一个固定点 (70, 40)
# -----------------------------
def add_endpoints(x_list: List[float], y_list: List[float],
                  lon: float = 70.0, lat: float = 40.0) -> Tuple[List[float], List[float]]:
    x_list = [lon] + x_list + [lon]
    y_list = [lat] + y_list + [lat]
    return x_list, y_list


# -----------------------------
# 3) 计算球面距离（公里）
#    使用 Haversine 公式，避免 arccos 域错误
#    半径 R = 6370 km
# -----------------------------
R_EARTH_KM = 6370.0

def get_dis(p1: Tuple[float, float], p2: Tuple[float, float], R: float = R_EARTH_KM) -> float:
    """
    p = (lon_deg, lat_deg)
    返回球面两点之间的大圆距离（公里）
    """
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    a = min(1.0, max(0.0, a))  # 数值稳定
    c = 2.0 * math.asin(math.sqrt(a))
    return R * c


def build_distance_matrix(xs: List[float], ys: List[float]) -> np.ndarray:
    n = len(xs)
    mat = np.zeros((n, n), dtype=float)
    points = [(xs[i], ys[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = get_dis(points[i], points[j], R_EARTH_KM)
            mat[i, j] = mat[j, i] = d
    return mat


# -----------------------------
# 4) 模拟退火求解：固定起点 index=0 和终点 index=n-1
#    中间的 1..n-2 这段做 2-opt 片段反转邻域搜索
# -----------------------------
def path_length(order: List[int], dist: np.ndarray) -> float:
    return sum(dist[order[i], order[i + 1]] for i in range(len(order) - 1))


def sa_tsp_with_fixed_endpoints(
    dist: np.ndarray,
    T0: float = 1000.0,
    Tmin: float = 1e-5,
    alpha: float = 0.99999,
    iters_per_T: int = 1000,
    max_no_improve_rounds: int = 200,
    rng_seed: int = 42,
) -> Tuple[float, List[int]]:
    """
    固定起点 0，终点 n-1，访问全部结点一次。
    使用 2-opt 反转邻域，仅在中间区间 [1, n-2] 上操作。
    """
    random.seed(rng_seed)
    n = dist.shape[0]

    # 初始解：0, 1, 2, ..., n-2, n-1
    order = list(range(n))
    best_order = order[:]
    best_len = path_length(order, dist)

    T = T0
    no_improve_rounds = 0

    while T > Tmin and no_improve_rounds < max_no_improve_rounds:
        improved_this_round = False

        for _ in range(iters_per_T):
            # 在 [1, n-2] 之间随机选两个索引，做片段反转（2-opt）
            if n <= 3:
                break  # 太短无法操作

            i = random.randint(1, n - 3)    # 1..n-3
            j = random.randint(i + 1, n - 2)  # i+1..n-2

            # 计算增量（只需看受影响的边）
            a, b = order[i - 1], order[i]
            c, d = order[j], order[j + 1]

            old_edges = dist[a, b] + dist[c, d]
            new_edges = dist[a, c] + dist[b, d]
            delta = new_edges - old_edges

            # Metropolis 接受准则
            if delta < 0 or random.random() < math.exp(-delta / T):
                # 接受：反转 [i, j]
                order[i:j + 1] = reversed(order[i:j + 1])
                curr_len = best_len + delta  # 使用增量更新
                # 记录当前长度（为了下一次增量参考，直接计算也可以）
                best_len = curr_len if curr_len < best_len else path_length(order, dist)

                # 如果更优则记录
                if best_len < path_length(best_order, dist):
                    best_order = order[:]
                    improved_this_round = True

        T *= alpha
        if improved_this_round:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

    # 最终结果
    final_len = path_length(best_order, dist)
    return final_len, best_order


# -----------------------------
# 5) 主流程：读取 -> 增加端点 -> 距离矩阵 -> SA -> 返回
# -----------------------------
def solve_tsp_sa(
    excel_path: str = "Simulated_annealing.xlsx",
    start_end_lon: float = 70.0,
    start_end_lat: float = 40.0,
) -> Tuple[float, List[int], np.ndarray, List[float], List[float]]:
    # 读取并添加端点
    xs, ys = load_points_from_excel(excel_path)
    xs, ys = add_endpoints(xs, ys, lon=start_end_lon, lat=start_end_lat)

    # 构建 102 x 102 矩阵（若原本 100 行）
    matrix_dis = build_distance_matrix(xs, ys)

    # 模拟退火（固定首尾）
    best_distance, best_order = sa_tsp_with_fixed_endpoints(matrix_dis)

    return best_distance, best_order, matrix_dis, xs, ys


if __name__ == "__main__":
    # 运行并打印结果
    best_distance_km, order, matrix_dis, x_list, y_list = solve_tsp_sa(
        excel_path="Simulated_annealing.xlsx",
        start_end_lon=70.0,
        start_end_lat=40.0,
    )

    print(f"最短总距离 (km): {best_distance_km:.6f}")
    print("遍历顺序的索引（含固定首尾）：")
    print(order)

    # 如需输出对应的经纬度顺序，可取消以下注释：
    # coords_ordered = [(x_list[i], y_list[i]) for i in order]
    # print(coords_ordered)


