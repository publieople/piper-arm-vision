#!/usr/bin/env python3
"""screwgrasp.py — 3D 打印大螺丝的跨轴夹取

两种入口:
  旧路径(默认, 螺丝在观察位正下方): --dry / --lift / --release / --lift-only
  新路径 --top : 从高位逐级"对中 -> 按轴旋转 -> 下降 -> 闭合 -> 抬升"
                 每一步都用 screwvision 的特征量闭环, 不用质心

用法:
  /home/po/arm-env/bin/python /home/po/vision/screwgrasp.py --top --dry     # 只对中+旋转, 不下降
  /home/po/arm-env/bin/python /home/po/vision/screwgrasp.py --top           # 到闭合
  /home/po/arm-env/bin/python /home/po/vision/screwgrasp.py --top --lift    # 再多抬 40mm 验证
  /home/po/arm-env/bin/python /home/po/vision/screwgrasp.py --top --release  # 放回+张开+回 home

实测数据(2026-11-05): 螺纹外径 59mm; 螺杆长 190mm(任何高度都装不进画面, 故不用质心);
  高位悬停 237mm 时 J5=69.7 贴死限位, +x 完全动不了 —— 高空悬停没必要, 直接低处对中;
  下降过程中 J5 从 67.9 一路松到 49.3; 抓取高度 z≈156mm, 闭合 55mm/6N 停在 60.2mm/-1.68N,
  抬 40mm 宽度 60.1 力 -1.65 (Δ<0.1) = 夹稳。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

import autograsp as A
import screwvision as SV

CFG = dict(
    obs_q=[-11.86, 92.52, -62.95, 0.0, 65.57, 0.0],
    j6=30.0,
    grip_z=165.0,          # 旧路径的抓取高度
    step=12.0,
    close_mm=55.0, close_n=6.0,
    open_mm=77.6, open_n=1.0,
    lift=40.0,
    shaft_mm=59.0,
    speed_down=6, speed_up=8,
)

CFG_TOP = dict(
    hover_z=200.0,         # 观察位高度: 228 时 J5 无余量, 转正后偏 127px 纠不动; 降到 200
    align_tol=15.0,        # 轴线垂直偏移容差(px)
    j6_limit=110.0,        # J6 自留 5 度余量(硬限位 115)
    grip_z=156.0,          # 实测夹得最稳的高度(螺纹外径 59, 闭合停在 60.2)
    step=20.0,             # 下降步长
    close_mm=55.0, close_n=6.0,
    open_mm=77.6, open_n=1.0,
    lift=40.0, lift_step=20.0,
    shaft_mm=59.0,
    speed_down=5, speed_up=6,
    probe_mm=55.0, probe_n=3.0,   # 物理探测: 闭合到 55mm / 3N, 看停在哪
    probe_span=12.0,              # 每次探测的 J6 间隔(度)
)


# ---------------------------------------------------------------- 旧路径
def find_screw(bgr, dep, tag="sc"):
    """黑色连通块 -> 轴中点像素/尺寸/角度(旧路径用, 依赖整块可见)"""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 2] < 90).astype(np.uint8) * 255
    if np.count_nonzero(dep > 0) > 0.05 * dep.size:
        m = cv2.bitwise_and(m, ((dep > 40) & (dep < 700)).astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def _blobs(mm):
        n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
        return sorted([(st[k, 4], k) for k in range(1, n) if st[k, 4] > 3000], reverse=True), lab, st

    cand, lab, st = _blobs(m)
    if not cand:
        cand, lab, st = _blobs((hsv[:, :, 2] < 90).astype(np.uint8) * 255)
    if not cand:
        return None
    k = cand[0][1]
    pts = cv2.findNonZero((lab == k).astype(np.uint8))
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    return dict(cx=cx, cy=cy, long=max(rw, rh), short=min(rw, rh),
                ang=ang if rw >= rh else ang + 90, area=int(st[k, 4]),
                depth=float(np.median(dep[lab == k])))


# ---------------------------------------------------------------- 新路径 --top
def settle(arm, dz, sp=5):
    """保持 xy/rpy, 只改 z"""
    p, rpy = A.pose(arm)
    return A.move_p(arm, [p[0], p[1], p[2] + dz, *rpy], speed=sp)


LAST_GAINS = None       # (gx,gy) 上一次现场标定的图像雅可比, 供转 J6 后复用
DEFAULT_GAINS = (-1.27, -4.12)   # 2026-11-08 实测可用的一组; 探针不可靠时兜底


def servo_xy(arm, tol=None, max_iter=14, gains=None):
    """把螺杆轴线推到画面中心: 用 axis_offset(垂直偏移), 现场探 x/y 两个方向"""
    global LAST_GAINS
    tol = tol if tol is not None else CFG_TOP["align_tol"]
    if gains is not None:
        gx, gy = gains
        print("[align] 沿用雅可比 x %+.2f  y %+.2f (不重探, 免走动)" % (gx, gy))
    else:
        bgr, _ = A.capture("top_a0")
        o0 = SV.axis_offset(bgr)
        if o0 is None:
            print("[align] 看不到螺丝"); return False
        p0, rpy = A.pose(arm)
        D = 8.0
        gains2 = []
        for i, axis in enumerate((1, 0)):          # 先 y 再 x
            tgt = [p0[0], p0[1], p0[2], *rpy]
            tgt[axis] += D
            ok, _ = A.move_p(arm, tgt, speed=5)
            bgr, _ = A.capture("top_aprobe%d" % i)
            o = SV.axis_offset(bgr) if ok else None
            A.move_p(arm, [p0[0], p0[1], p0[2], *rpy], speed=5)
            gains2.append(None if o is None else (o - o0) / D)
        gx, gy = gains2[1], gains2[0]
        mag = 0.0 if gx is None or gy is None else float(np.hypot(gx, gy))
        print("[align] 探针: x %s  y %s  |g|=%.2f px/mm"
              % ("-" if gx is None else "%+.2f" % gx, "-" if gy is None else "%+.2f" % gy, mag))
        if mag < 0.3 or mag > 8.0:
            # 探针只有 8mm 行程, 1D 偏移量的方向对噪声敏感, 实测同一天两次能差 3 倍。
            # 不再因此中止: 退回一组实测可用的固定增益, 靠后面的偏移突变检查兜底。
            gx, gy = DEFAULT_GAINS
            print("[align] 增益不可信(|g|=%.2f), 改用固定增益 x %+.2f y %+.2f"%(mag,gx,gy))
        LAST_GAINS = (gx, gy)
    prev = None
    best = 1e9
    for it in range(max_iter):
        bgr, _ = A.capture("top_a%d" % (it + 1))
        o = SV.axis_offset(bgr)
        p, rpy = A.pose(arm)
        j5 = A.cur_joints()[4]
        print("[align %d] 偏移 %s px  J5 %.1f" % (it, "None" if o is None else "%+.0f" % o, j5))
        if o is None:
            return False
        if abs(o) < tol:
            print("[align] 已对中(%.0f px)" % o); return True
        if prev is not None and abs(o - prev) > 60 and abs(o) > abs(prev) * 2:
            print("[align] 偏移突跳 %+.0f px, 重测一次" % (o - prev))
            bgr, _ = A.capture("top_achk")
            o2 = SV.axis_offset(bgr)
            if o2 is None or abs(o2 - prev) > 60:
                print("[align] 测量不可靠(可能夹爪剪影混入), 停"); return False
            o = o2
        best = min(best, abs(o))
        if abs(o) > max(best * 1.4, 40.0):
            print("[align] 偏移在变大(最佳 %.0f px), 停" % best); return False
        prev = o
        s = -o / (gx * gx + gy * gy) * 0.7      # 阻尼, 实测探针增益偏小会超调
        sx, sy = s * gx, s * gy
        sn = float(np.hypot(sx, sy))
        if sn > 8.0:
            sx, sy = sx * 8.0 / sn, sy * 8.0 / sn
        ok, _ = A.move_p(arm, [p[0] + sx, p[1] + sy, p[2], *rpy], speed=5)
        if not ok:
            if j5 > 69.0:                      # 腕部饱和: 往里收一点换余量
                r = max(1.0, float(np.hypot(p[0], p[1])))
                print("   J5 到限位, 内收 8mm 再试")
                A.move_p(arm, [p[0] - 8 * p[0] / r, p[1] - 8 * p[1] / r, p[2], *rpy], speed=5)
                continue
            print("[align] 走不动"); return False
    return False


def angle_any(bgr):
    """优先用细长连通域(螺杆), 拿不到再退回 dt 剔螺帽的整块 PCA"""
    r = SV._angle_thin(bgr)
    if r is not None:
        return r[0], r[1]
    return SV.shaft_angle(bgr)


def rotate_j6(arm, tol=6.0, max_iter=6):
    """转 J6 让螺杆在画面里竖起来 => 夹爪开口横跨轴线"""
    import math
    bgr, _ = A.capture("top_r0")
    a0, _ = angle_any(bgr)
    if a0 is None:
        print("[yaw] 看不到螺丝"); return False
    j6_0 = A.cur_joints()[5]
    A.move_j(arm, [*A.cur_joints()[:5], j6_0 - 10.0], speed=5, timeout=25)
    bgr, _ = A.capture("top_r1")
    a1, _ = angle_any(bgr)
    if a1 is None:
        return False
    g = SV.wrap(a1 - a0, 180.0) / -10.0
    if abs(g) < 0.3:
        g = 1.1
    print("[yaw] 起始 %.0f deg, 探针增益 %.2f deg/deg, J6 %.1f" % (a1, g, A.cur_joints()[5]))
    cur = a1
    gsign = 1.0 if g >= 0 else -1.0        # 增益符号由首测定死, 不许中途翻向(否则原地打转)
    for it in range(max_iter):
        diff = SV.wrap(90.0 - cur, 180.0)
        print("[yaw %d] 轴向角 %.0f deg, 差 %+.0f deg, J6 %.1f" % (it, cur, diff, A.cur_joints()[5]))
        if abs(diff) < tol:
            print("[yaw] 夹爪开口已垂直螺杆"); return True
        dj = max(-20.0, min(20.0, diff / g))
        j6 = A.cur_joints()[5]
        if abs(j6 + dj) > CFG_TOP["j6_limit"]:
            dj = math.copysign(max(1.0, CFG_TOP["j6_limit"] - abs(j6)), dj)
            print("   J6 到自留限位, 改 %.1f deg" % dj)
        A.move_j(arm, [*A.cur_joints()[:5], j6 + dj], speed=5, timeout=25)
        bgr, _ = A.capture("top_r%d" % (it + 2))
        nxt, _ = angle_any(bgr)
        if nxt is None:                    # 画面结构变了, 微调一下再测
            for _k in range(2):
                j6b = A.cur_joints()[5] + 12.0
                A.move_j(arm, [*A.cur_joints()[:5], j6b], speed=5, timeout=25)
                bgr, _ = A.capture("top_rx")
                nxt, _ = angle_any(bgr)
                if nxt is not None:
                    break
            if nxt is None:
                print("[yaw] 测不到轴向角"); return False
        da, dq = SV.wrap(nxt - cur, 180.0), A.cur_joints()[5] - j6
        if abs(dq) > 2 and abs(da) > 2:
            g_new = da / dq
            if 0.5 < abs(g_new) < 1.6 and g_new * gsign > 0:   # 物理增益约 1, 越界/翻向说明画面被挡
                g = g_new
                print("   实测增益 %.2f deg/deg" % g)
            else:
                print("   增益 %.2f 不可信(画面被挡), 沿用 %.2f" % (g_new, g))
        cur = nxt
    return abs(SV.wrap(90.0 - cur, 180.0)) < tol


# ------------------------------------------- 物理对轴: 用夹爪"摸"出正确的 J6
def probe_width(arm, eff):
    """降到抓取高度轻夹一次 -> 读闭合宽度 -> 张开 -> 升回悬停高度

    返回 (宽mm, 力N, 是否真的夹到东西)。宽 ≈ 指令宽 说明两指之间没东西。
    """
    p, rpy = A.pose(arm)
    z0 = p[2]
    dz = z0 - CFG_TOP["grip_z"]
    if dz > 1.0:
        n = max(1, int(np.ceil(dz / CFG_TOP["step"])))
        for i in range(1, n + 1):
            ok, _ = A.move_p(arm, [p[0], p[1], z0 - dz * i / n, *rpy],
                             speed=CFG_TOP["speed_down"], timeout=10)
            if not ok:
                print("  [probe] 下降受阻")
                settle(arm, CFG_TOP["hover_z"] - A.pose(arm)[0][2], sp=CFG_TOP["speed_up"])
                return None
    w, f = A.gripper(eff, CFG_TOP["probe_mm"], CFG_TOP["probe_n"], 1.6)
    A.gripper(eff, CFG_TOP["open_mm"], CFG_TOP["open_n"], 0.9)
    settle(arm, CFG_TOP["hover_z"] - A.pose(arm)[0][2], sp=CFG_TOP["speed_up"])
    return w, f


def find_jaw_angle(arm, eff, max_round=3):
    """物理找 J6: 两指跨着螺杆闭合时, 闭合宽度 = 螺杆直径 / cos(偏角)

    所以"宽度最小"就是"开合方向垂直于螺杆轴"。宽-角度曲线是单峰近抛物线,
    三点探测 + 抛物线顶点迭代, 一般 3~6 次探测收敛。不依赖任何图像特征。
    """
    def at(j6):
        q = A.cur_joints()
        ok, _ = A.move_j(arm, [*q[:5], float(j6)], speed=6, timeout=20)
        if not ok:
            print("  [j6] 转 J6=%.1f 失败" % j6)
            return None
        r = probe_width(arm, eff)
        if r is None:
            return None
        w, f = r
        contact = w > CFG_TOP["probe_mm"] + 0.6
        print("[probe] J6 %+6.1f  宽 %5.1fmm 力 %+5.2fN  %s"
              % (j6, w, f, "接触" if contact else "指间空(没夹到)"))
        return w if contact else None

    j = A.cur_joints()[5]
    h = CFG_TOP["probe_span"]
    for rd in range(max_round):
        pts = []
        for d in (-h, 0.0, h):
            w = at(j + d)
            if w is not None:
                pts.append((j + d, w))
        if not pts:
            print("[j6] 三次探测都没碰到螺杆(先把它对到画面中心)"); return None
        if len(pts) < 2:
            j = pts[0][0]; break
        cur = min(pts, key=lambda p: p[1])[0]
        if len(pts) == 3:
            (x1, y1), (x2, y2), (x3, y3) = pts
            den = (x1 - x2) * (x1 - x3) * (x2 - x3)
            a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / den
            b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / den
            v = -b / (2 * a) if abs(a) > 1e-9 else cur
            v = cur + float(np.clip(v - cur, -h, h))    # 顶点只允许外推一段, 防止跳飞
        else:
            v = cur
        print("[j6] 第%d轮 顶点 %.1f (最小宽 %.1fmm)" % (rd + 1, v, min(p[1] for p in pts)))
        j = v
        if abs(j) > CFG_TOP["j6_limit"]:
            print("[j6] 顶点 %.1f 超出 ±%.0f 限位, 放弃" % (j, CFG_TOP["j6_limit"]))
            return None
        if abs(v - cur) < 2.0:
            break
        h = max(6.0, h / 2.0)
    q = A.cur_joints()
    A.move_j(arm, [*q[:5], float(j)], speed=5, timeout=20)
    w = at(j)
    if w is None:
        print("[j6] 最终位置没夹到东西, 放弃"); return None
    print("[j6] 对齐 J6 = %.1f, 闭合宽 %.1fmm (螺杆 %.0fmm)" % (j, w, CFG_TOP["shaft_mm"]))
    return j


# ------------------------------------------- 自动转角 B: J6 粗扫 + 各向异性评分
def aniso(bgr, smax=90, vmax=105):
    """画面里黑件像素的 纵向二阶矩 / 横向二阶矩。

    螺丝横放时 x 铺得开 (r<1), 竖直时 y 铺得开 (r>1.5)。
    物体是否出画、螺帽有没有被夹爪挡住都不影响: 只用二阶矩, 不做连通域/轮廓。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    m = ((hsv[:, :, 1] < smax) & (hsv[:, :, 2] < vmax)).astype(np.uint8)
    m[:20, :] = 0
    m[430:, :] = 0
    m[:, :60] = 0
    m[:, 600:] = 0
    ys, xs = np.nonzero(m)
    if xs.size < 2000:
        return None
    xm, ym = xs.mean(), ys.mean()
    sxx = ((xs - xm) ** 2).mean()
    syy = ((ys - ym) ** 2).mean()
    return float(syy / max(sxx, 1e-6))


def find_yaw_scan(arm, coarse=15, fine=5):
    """粗扫 J6(-90..+90) 找 r 的峰, 再在峰两侧细扫。返回 (j6, r)。"""
    q0 = A.cur_joints()
    best = None
    for j in range(-90, 91, coarse):
        ok, _ = A.move_j(arm, [*q0[:5], float(j)], speed=6, timeout=25)
        if not ok:
            print("  [scan] J6 %+d 转不过去" % j)
            continue
        bgr, _ = A.capture("sc%+d" % j)
        r = aniso(bgr)
        if r is None:
            print("  [scan] J6 %+4d 无黑件" % j)
            continue
        print("  [scan] J6 %+4d  r=%.3f" % (j, r))
        if best is None or r > best[1]:
            best = (float(j), r)
    if best is None:
        print("[yaw] 粗扫没有可用画面")
        return None
    jb = best[0]
    for j in (jb - 2 * fine, jb - fine, jb + fine, jb + 2 * fine):
        if abs(j) > 105:
            continue
        ok, _ = A.move_j(arm, [*q0[:5], float(j)], speed=5, timeout=25)
        if not ok:
            continue
        bgr, _ = A.capture("sf%+d" % int(j))
        r = aniso(bgr)
        if r is None:
            continue
        print("  [fine] J6 %+5.1f  r=%.3f" % (j, r))
        if r > best[1]:
            best = (float(j), r)
    print("[yaw] 自动转角 J6 = %+.1f (r=%.2f, 峰=竖直)" % best)
    return best


def run_top(a):
    arm, eff = A.connect()
    if a.release:
        p, rpy = A.pose(arm)
        settle(arm, CFG_TOP["grip_z"] - p[2], sp=5)
        print("[release] 张开", A.gripper(eff, CFG_TOP["open_mm"], CFG_TOP["open_n"], 1.5))
        settle(arm, CFG_TOP["lift"], sp=6)
        A.goto_q(arm, A.CFG["home_q"], steps=4, speed=12)
        print("[home]", np.round(A.pose(arm)[0], 1), np.round(A.pose(arm)[1]))
        arm.disconnect(); return 0

    # 1) 到观察位, 夹爪张开
    A.goto_q(arm, CFG["obs_q"], steps=4, speed=10)
    A.gripper(eff, CFG_TOP["open_mm"], CFG_TOP["open_n"], 1.2)
    print("[obs] 法兰", np.round(A.pose(arm)[0], 1), "J5 %.1f" % A.cur_joints()[4])
    p_obs, _ = A.pose(arm)
    if p_obs[2] - CFG_TOP["hover_z"] > 3.0:        # 228 高度 J5 没余量, 先降下来
        settle(arm, CFG_TOP["hover_z"] - p_obs[2], sp=6)
        print("[hover] 降到 %.0fmm, J5 %.1f" % (A.pose(arm)[0][2], A.cur_joints()[4]))

    # 2) 对中 -> 2.5) 沿螺杆走开(让螺帽出画) -> 3) 按轴旋转
    if not servo_xy(arm):
        print("[fail] 对中失败"); arm.disconnect(); return 2
    if a.j6 is not None:
        q = A.cur_joints()
        A.move_j(arm, [*q[:5], float(a.j6)], speed=5, timeout=25)
        print("[yaw] 直接用指定 J6 %.1f" % A.cur_joints()[5])
    else:
        jf = find_yaw_scan(arm)
        if jf is None:
            print("[fail] 找夹持角失败"); arm.disconnect(); return 3
    if not servo_xy(arm, tol=20.0, gains=LAST_GAINS):   # 转完画面会平移, 再对一次
        print("[fail] 旋转后对中失败"); arm.disconnect(); return 2

    # 4) 下降前预检可达性
    p, rpy = A.pose(arm)
    plan = [p[0], p[1], CFG_TOP["grip_z"], *rpy]
    info = A.kin.check(plan[:3], plan[3:], seed=A.cur_joints())
    if not info["ok"]:
        print("[fail] 抓取位姿不可达: %s" % info["msg"]); arm.disconnect(); return 4
    print("[plan] 抓取位姿可达 err=%.1fmm 最紧余量 %.1f deg" % (info["err"], info["worst"]))
    if a.dry:
        print("[dry] 不下降, 到此为止"); arm.disconnect(); return 0

    # 5) 分段下降
    p, rpy = A.pose(arm)
    dz = p[2] - CFG_TOP["grip_z"]
    n = max(1, int(np.ceil(dz / CFG_TOP["step"])))
    for i in range(1, n + 1):
        tgt = p[2] - dz * i / n
        ok, err = A.move_p(arm, [p[0], p[1], tgt, *rpy], speed=CFG_TOP["speed_down"], timeout=10)
        print("  down %d/%d z=%.0f ok=%s J5 %.1f" % (i, n, A.pose(arm)[0][2], ok, A.cur_joints()[4]))
        if not ok:
            print("[fail] 下降受阻, 停"); arm.disconnect(); return 5

    # 6) 闭合 + 判定
    w, f = A.gripper(eff, CFG_TOP["close_mm"], CFG_TOP["close_n"], 2.5)
    print("[grip] 宽 %.1fmm 力 %.2fN" % (w, f))
    if not (w > CFG_TOP["shaft_mm"] - 8 and abs(f) > 1.0):
        print("[fail] 没夹住(空夹), 张开退出")
        A.gripper(eff, CFG_TOP["open_mm"], CFG_TOP["open_n"], 1.2)
        arm.disconnect(); return 6

    # 7) 抬升验证
    if a.lift:
        for i in (1, 2):
            settle(arm, CFG_TOP["lift_step"], sp=CFG_TOP["speed_up"])
            w2, f2 = A.gripper(eff, CFG_TOP["close_mm"], CFG_TOP["close_n"], 0.8)
            print("[lift %d] z=%.0f 宽 %.1fmm 力 %.2fN" % (i, A.pose(arm)[0][2], w2, f2))
            if abs(w2 - w) > 3.0:
                print("[fail] 宽度变化 >5mm, 螺丝在滑"); arm.disconnect(); return 7
    print("[done] 抓取成功")
    arm.disconnect(); return 0






def run_legacy(a):
    arm, eff = A.connect()
    if a.lift_only:
        w, f = A.gripper(eff, CFG["close_mm"], CFG["close_n"], 1.5)
        print("[hold] 宽度 %.1fmm 力 %.2fN" % (w, f))
        if not (w > CFG["shaft_mm"] - 8 and abs(f) > 1.0):
            print("[fail] 当前没夹住, 先跑一次不带参数的")
            arm.disconnect(); sys.exit(2)
        p, rpy = A.pose(arm)
        for i in (1, 2):
            A.move_p(arm, [p[0], p[1], p[2] + CFG["lift"] / 2 * i, *rpy], speed=5, timeout=10)
            w2, f2 = A.gripper(eff, CFG["close_mm"], CFG["close_n"], 0.8)
            print("[lift] %d z=%.0f 宽 %.1fmm 力 %.2fN" % (i, A.pose(arm)[0][2], w2, f2))
            if abs(w2 - w) > 3.0:
                print("[fail] 宽度变化 >5mm, 螺丝在滑"); arm.disconnect(); sys.exit(3)
        print("[done] 抬升 %0.fmm 未滑, 抓取成功" % CFG["lift"])
        arm.disconnect(); return 0

    if a.release:
        p, rpy = A.pose(arm)
        A.move_p(arm, [p[0], p[1], CFG["grip_z"], *rpy], speed=5, timeout=10)
        print("[release] 张开", A.gripper(eff, CFG["open_mm"], CFG["open_n"], 1.5))
        p, rpy = A.pose(arm)
        A.move_p(arm, [p[0], p[1], 215.0, *rpy], speed=8, timeout=10)
        A.goto_q(arm, A.CFG["home_q"], steps=4, speed=12)
        print("[home]", np.round(A.pose(arm)[0], 1), np.round(A.pose(arm)[1]))
        arm.disconnect(); return 0

    A.goto_q(arm, CFG["obs_q"], steps=4, speed=10)
    q = A.cur_joints()
    A.goto_q(arm, [*q[:5], CFG["j6"]], steps=1, speed=8)
    print("[obs] 法兰", np.round(A.pose(arm)[0], 1), "J6 %.0f" % A.cur_joints()[5])

    bgr, dep = A.capture("sc_grip")
    s = find_screw(bgr, dep)
    if s is None:
        sys.exit("[fail] 视野里没看到螺丝")
    print("[see] 像素(%.0f,%.0f) %.0fx%.0f ang %.1f depth %.0f"
          % (s["cx"], s["cy"], s["long"], s["short"], s["ang"], s["depth"]))
    dpx = np.hypot(s["cx"] - 320, s["cy"] - 240)
    print("[see] 距画面中心 %.0f px" % dpx)
    if dpx > 90 and not a.force:
        sys.exit("[abort] 螺丝距画面中心 %.0fpx > 90px, 没对准不做下降. 先摆正, 或加 --force" % dpx)
    if a.dry:
        print("[dry] 不做动作"); arm.disconnect(); return 0

    A.gripper(eff, CFG["open_mm"], CFG["open_n"], 1.2)
    p, rpy = A.pose(arm)
    dz = p[2] - CFG["grip_z"]
    n = max(1, int(np.ceil(dz / CFG["step"])))
    for i in range(1, n + 1):
        A.move_p(arm, [p[0], p[1], p[2] - dz * i / n, *rpy], speed=CFG["speed_down"], timeout=10)
        print("  down %d/%d z=%.0f" % (i, n, A.pose(arm)[0][2]))
    w, f = A.gripper(eff, CFG["close_mm"], CFG["close_n"], 2.5)
    print("[grip] 宽度 %.1fmm 力 %.2fN" % (w, f))
    if not (w > CFG["shaft_mm"] - 8 and abs(f) > 1.0):
        print("[fail] 没夹住(空夹), 张开退出")
        A.gripper(eff, CFG["open_mm"], CFG["open_n"], 1.0)
        arm.disconnect(); return 2
    if a.lift:
        p, rpy = A.pose(arm)
        for i in (1, 2):
            A.move_p(arm, [p[0], p[1], p[2] + CFG["lift"] / 2 * i, *rpy], speed=5, timeout=10)
            w2, f2 = A.gripper(eff, CFG["close_mm"], CFG["close_n"], 0.8)
            print("[lift] %d z=%.0f 宽 %.1fmm 力 %.2fN" % (i, A.pose(arm)[0][2], w2, f2))
            if abs(w2 - w) > 3.0:
                print("[fail] 宽度变化 >5mm, 螺丝在滑"); break
    print("[done] 抓取成功")
    arm.disconnect(); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", action="store_true", help="新路径: 对中->按轴旋转->下降->闭合(推荐)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--lift", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lift-only", action="store_true")
    ap.add_argument("--record", nargs="?", const="auto", default=None)
    ap.add_argument("--j6", type=float, default=None, help="直接指定 J6(度), 跳过自动转角")
    a = ap.parse_args()
    if a.record:
        import os, time
        A.REC_PATH = (os.path.join(A.OUT, "run_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
                      if a.record == "auto" else a.record)
        print("[record] %s" % A.REC_PATH)
    return run_top(a) if a.top else run_legacy(a)


def head_stat(bgr):
    """返回 (质心 px, 螺帽中心 px, 最厚处半径 dt, 面积) , 取不到返回 None"""
    import cv2
    m = SV.dark_mask(bgr, jaw_cut=0.78)
    r = SV.biggest(m, dilate=3)
    if r is None:
        return None
    lab, st, ki = r
    blob = (lab == ki).astype(np.uint8)
    ys, xs = np.where(blob > 0)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    hy, hx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    c = np.array([xs.mean(), ys.mean()])
    return c, np.array([hx, hy], float), float(dt[hy, hx]), int(len(xs))


def probe_jac(arm, D=8.0):
    """现场标定图像雅可比 J: d(暗区质心 px)/d(法兰 mm), 2x2"""
    p0, rpy = A.pose(arm)
    bgr, _ = A.capture("top_j0")
    h0 = head_stat(bgr)
    if h0 is None:
        return None
    c0 = h0[0]
    cols = []
    for axis in (1, 0):
        tgt = [p0[0], p0[1], p0[2], *rpy]
        tgt[axis] += D
        ok, _ = A.move_p(arm, tgt, speed=5)
        bgr, _ = A.capture("top_jp%d" % axis)
        A.move_p(arm, [p0[0], p0[1], p0[2], *rpy], speed=5)
        h = head_stat(bgr) if ok else None
        if h is None:
            return None
        cols.append((h[0] - c0) / D)
    J = np.stack([cols[0], cols[1]], 1)      # [d/dy, d/dx]
    return np.array([[J[0, 1], J[0, 0]], [J[1, 1], J[1, 0]]])   # 列顺序 x,y


def walk_off_head(arm, steps=3, px=170.0):
    """沿螺杆方向把相机往前挪, 直到螺帽(与垫子)退出画面

    昨天的成功全靠这一步: 画面里只剩螺杆时 PCA 才稳(94 deg, 真值 91)。
    方向不需要精确: 用"暗区最厚处=螺帽中心"和整体质心之差, 精度 ±40 度就够。
    """
    J = probe_jac(arm)
    if J is None:
        print("[walk] 雅可比标不出来"); return False
    print("[walk] 雅可比 [[%.2f %.2f],[%.2f %.2f]] px/mm" % (J[0, 0], J[0, 1], J[1, 0], J[1, 1]))
    for i in range(steps):
        bgr, _ = A.capture("top_w%d" % i)
        h = head_stat(bgr)
        if h is None:
            print("[walk] 看不到螺丝"); return False
        c, hd, dmax, area = h
        u = c - hd
        u = u / (np.linalg.norm(u) + 1e-9)
        print("[walk %d] 面积 %d 最厚 %.0f px 方向 (%+.2f,%+.2f)" % (i, area, dmax, u[0], u[1]))
        if dmax < 42.0 and area < 40000:
            print("[walk] 螺帽已出画, 只剩螺杆"); return True
        try:
            dxy = np.linalg.solve(J, u * px)
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(dxy)):
            return False
        n = float(np.hypot(dxy[0], dxy[1]))
        if n > 60.0:
            dxy = dxy * 60.0 / n
        p, rpy = A.pose(arm)
        ok, _ = A.move_p(arm, [p[0] + dxy[0], p[1] + dxy[1], p[2], *rpy], speed=5)
        print("   走 (%.1f, %.1f) mm  ok=%s  J5 %.1f" % (dxy[0], dxy[1], ok, A.cur_joints()[4]))
        if not ok:
            print("[walk] 走不动, 停"); return False
    bgr, _ = A.capture("top_wend")
    h = head_stat(bgr)
    return bool(h is not None and h[2] < 45.0)


if __name__ == "__main__":
    sys.exit(main())
