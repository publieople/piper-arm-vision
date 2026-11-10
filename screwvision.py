#!/usr/bin/env python3
"""screwvision.py — 螺丝的视觉特征(全部不依赖质心)

为什么不用质心: 螺丝长 190mm, 相机在任何可达高度都装不下整根,
黑块永远贴边, 质心随裁剪乱跳(实测伺服 102→112→119→89→52 不收敛)。

提供的量:
  axis_offset  轴线到画面中心的**垂直**偏移(px) —— 沿轴向被裁掉也不影响
  shaft_angle  轴线在画面里的角度(deg, 0=水平, 90=竖直)
  left_edge    深色区最左一列(px) —— 沿轴向的标尺(会被推出画面, 只作短程用)
  e2 投影:     把螺杆轴线投到画面中心所需的横移, 用现场探到的 px/mm 换算
"""
import cv2
import numpy as np


# 分析窗: 只保留画面中央这块。夹爪指板是同一片黑色, 转 J6 时它会在画面里绕圈,
# 底部固定条带(旧 jaw_cut)挡不住 —— 实测转完 J6 后指板插进来把"轴线中点"整体拉偏,
# 偏移从 -21px 直接跳到 -158px, 伺服当场发散。
ROI_X = (0.10, 0.90)
ROI_Y = (0.02, 0.70)


def dark_mask(bgr, jaw_cut=0.78, v_max=90, roi=False, margin=0):
    """V<90 的暗区; 默认砍掉底部 22%(夹爪指板)

    roi=True 时只保留中央窗口 —— 用于**轴线偏移**: 转 J6 后指板会绕到画面里,
    把"轴线中点"整体拉偏(实测 -21 -> -158px)。角度估计仍用全幅, 裁剪会让
    主轴拟合变噪(实测增益在 2.8 与 -1.3 之间乱跳)。
    """
    if bgr is None:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 2] < v_max).astype(np.uint8)
    h, w = m.shape[:2]
    if roi:
        box = np.zeros_like(m)
        x0 = int(w * ROI_X[0]) + margin
        x1 = int(w * ROI_X[1]) - margin
        box[int(h * ROI_Y[0]):int(h * ROI_Y[1]),
            max(0, x0):min(w, x1)] = 1
        m = (m * box).astype(np.uint8)
    if jaw_cut is not None and not roi:
        m[int(h * jaw_cut):, :] = 0
    return m


def biggest(mask, min_area=1500, dilate=7):
    if mask is None:
        return None
    m = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    if st[k, 4] < min_area:
        return None
    return lab, st, k


def shaft_pts(bgr, roi=False, kdt=1.4):
    """螺杆像素点 Nx2 (只在螺帽以外的部分), 找不到返回 None

    螺帽是整块暗区里最"厚"的部分: distanceTransform 的最大值点就是它的中心,
    剔除半径 kdt*max_dt 内的像素后剩下的基本只有螺杆。
    """
    m = dark_mask(bgr, roi=roi)
    r = biggest(m)
    if r is None:
        return None
    lab, st, ki = r
    blob = (lab == ki).astype(np.uint8)
    ys, xs = np.where(blob > 0)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    hy, hx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    hmax = float(dt[hy, hx])
    keep = ((xs - hx) ** 2 + (ys - hy) ** 2) > (kdt * hmax) ** 2
    if int(keep.sum()) < 1200:            # 剔太狠就退回全量
        keep = np.ones(len(xs), bool)
    return np.stack([xs[keep], ys[keep]], 1).astype(np.float64)


def axis_offset(bgr):
    """轴线垂直偏移 px: >0 / <0 表示轴线在中心的两侧; 找不到返回 None"""
    X = shaft_pts(bgr, roi=True)
    if X is None:
        return None
    a = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    if a[1] < 0 or (a[1] == 0 and a[0] < 0):   # SVD 主轴方向任意, 统一成"朝下", 否则偏移符号会翻
        a = -a
    n = np.array([-a[1], a[0]])
    proj = X @ n
    # 用投影均值(=质心在法向上的投影)而不是 min/max 中点: 后者对掩膜里一两个
    # 离群像素极其敏感, 实测同一姿态下 8mm 平移的增益会在 2.8 与 9.9 px/mm 之间跳,
    # 导致伺服震荡。均值参与点上千个, 重复性实测 ±0.5px。
    mid = float(proj.mean())
    c = np.array([bgr.shape[1] / 2.0, bgr.shape[0] / 2.0]) @ n
    return float(mid - c)          # 轴线相对画面中心的垂直偏移(px)


def shaft_angle(bgr, kdt=1.4):
    """轴线角度 deg mod 180 (0=水平, 90=竖直), 以及暗区面积

    先 distanceTransform 找暗区最厚处(=螺帽中心), 剔除其 kdt*max_dt 半径内的像素,
    剩下基本只有螺杆, 再 PCA。直接对整块 L 形 PCA 会被螺帽拉斜(实测 122° vs 真值 95°)。
    """
    X = shaft_pts(bgr, roi=False, kdt=kdt)
    if X is None:
        return None, 0
    area = int(len(X))
    X = X - X.mean(0)
    a = np.linalg.svd(X, full_matrices=False)[2][0]
    ang = float(np.degrees(np.arctan2(a[1], a[0])) % 180.0)
    return ang, area


def left_edge(bgr, min_px=10):
    """暗区最左一列(px), 用作沿轴向的短程标尺"""
    m = dark_mask(bgr)
    if m is None:
        return None
    cols = np.where(m.sum(0) >= min_px)[0]
    if len(cols) == 0:
        return None
    return float(cols.min())


def wrap(a, mod=180.0):
    """把角度误差折到 (-mod/2, mod/2]

    螺杆轴向角是 mod 180 的周期量: 90=竖直, 0/180=水平。
    (用 mod 90 会把 179 误判成"离竖直只差 1 度" —— 踩过)
    """
    a = a % mod
    if a > mod / 2:
        a -= mod
    return a


def _angle_thin(bgr, jaw_cut=0.78, vmax=85, thin=(14, 46), amin=9000):
    """只取「细长且不贴边」的连通域(螺杆本体)做主成分

    灰色桌垫与螺丝同属暗区, 螺帽摆到左侧时三者会连成一块, 主轴被垫子拉斜
    (实测同一帧 25°/137° 两个分支来回跳)。螺杆自身很细(dt 最大 27~33px)、
    也不贴画面边, 用这两个条件能把它单独挑出来。
    """
    if bgr is None:
        return None
    H, W = bgr.shape[:2]
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = (V < vmax).astype(np.uint8)
    m[int(H * jaw_cut):, :] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for k in range(1, n):
        a = int(st[k, 4])
        if a < amin:
            continue
        x, y, w, h, _ar = st[k]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                      # 贴边 -> 是垫子
        comp = (lab == k).astype(np.uint8)
        dmax = float(cv2.distanceTransform(comp, cv2.DIST_L2, 5).max())
        if not (thin[0] <= dmax <= thin[1]):
            continue
        ys, xs = np.where(comp > 0)
        X = np.stack([xs, ys], 1).astype(np.float64)
        X -= X.mean(0)
        v = np.linalg.svd(X, full_matrices=False)[2][0]
        ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
        if best is None or a > best[0]:
            best = (a, ang)
    return None if best is None else (best[1], best[0])

