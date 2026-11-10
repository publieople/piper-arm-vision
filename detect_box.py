#!/usr/bin/env python3
"""在 RGB-D 帧里找"方形纸盒"，输出相机坐标系下的 3D 位姿（mm / 度）。

思路（经典几何，不训练）：
  1. HSV 阈值取牛皮纸盒的棕色 -> 掩码
  2. 形态学开闭 + 连通域，按面积/长宽比/实心度筛
  3. 对最佳轮廓做 minAreaRect，得到像素中心、边长、朝向
  4. 用对齐后的深度在轮廓内取中值 -> Z；结合内参反投影得到 (X,Y,Z)
  5. 用 Z 和像素尺寸反算盒子真实尺寸（mm），输出 yaw

用法:
  /home/po/vision-env/bin/python detect_box.py --tag box [--min-area 2000] [--show]
输入: out/<tag>_color.png / out/<tag>_depth.npy
输出: out/<tag>_detect.png（标注图）、stdout JSON
"""
import argparse
import json
import os

import cv2
import numpy as np

OUT = "/home/po/vision/out"


def load_intrinsics():
    """优先用标定文件；没有就用 SDK 报的（cy 可疑，标定前只作参考）。"""
    f = os.path.join(OUT, "camera_intrinsics.npz")
    if os.path.exists(f):
        d = np.load(f)
        return dict(fx=float(d["fx"]), fy=float(d["fy"]), cx=float(d["cx"]),
                    cy=float(d["cy"]), source="calibrated:" + f)
    return dict(fx=488.83, fy=488.83, cx=314.65, cy=214.66, source="sdk-default(suspect)")


def cardboard_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 2026-11-04 放宽：原 (8,40,60)-(32,210,235) 在阴影/白标签下只截到半块盒子（面积 4114, ar 1.93）
    m = cv2.inRange(hsv, (5, 30, 40), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="box")
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--max-area", type=int, default=200000)
    ap.add_argument("--min-depth", type=int, default=250)
    ap.add_argument("--max-depth", type=int, default=1500)
    args = ap.parse_args()

    color = cv2.imread(f"{OUT}/{args.tag}_color.png")
    depth = np.load(f"{OUT}/{args.tag}_depth.npy")
    if color is None or depth is None:
        raise SystemExit("缺少输入，先跑 capture.py")
    K = load_intrinsics()
    print("intrinsics:", K)

    mask = cardboard_mask(color)
    # 深度门限：只保留 0.25~1.5 m 的东西（抓取工作区）
    dgate = ((depth > args.min_depth) & (depth < args.max_depth)).astype(np.uint8) * 255
    dgate = cv2.dilate(dgate, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(mask, dgate)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = depth.shape
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < args.min_area or a > args.max_area:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        if min(w, h) < 12:
            continue
        ar = max(w, h) / max(min(w, h), 1e-6)
        solidity = a / max(w * h, 1e-6)
        cm = np.zeros(depth.shape, np.uint8)
        cv2.drawContours(cm, [c], -1, 255, -1)
        dv = depth[(cm > 0) & (depth > 0)]
        if dv.size < 80:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        touches = (x <= 1 or y <= 1 or x + bw >= w_img - 1 or y + bh >= h_img - 1)
        cands.append(dict(c=c, rect=rect, a=a, ar=ar, sol=solidity, z=float(np.median(dv)),
                          touches=touches, dv=dv.size))
    cands.sort(key=lambda k: (k["z"], -k["a"]))  # 最近的优先（抓取目标通常是近处物体）
    best = None
    for k in cands:
        if 0.55 <= k["ar"] <= 1.9 and k["sol"] > 0.30:
            best = k
            break
    print("candidates:", json.dumps([{k: (round(v, 1) if isinstance(v, float) else v)
                                      for k, v in c.items() if k in ("a", "ar", "sol", "z", "touches")}
                                     for c in cands], ensure_ascii=False))

    if best is None:
        print(json.dumps({"found": False, "n_contours": len(cnts)}))
        return

    c, rect, area, ar, sol = best["c"], best["rect"], best["a"], best["ar"], best["sol"]
    (u, v), (w, h), ang = rect
    cmask = np.zeros(depth.shape, np.uint8)
    cv2.drawContours(cmask, [c], -1, 255, -1)
    dvals = depth[(cmask > 0) & (depth > 0)]
    if dvals.size == 0:
        print(json.dumps({"found": False, "reason": "no depth inside contour"}))
        return
    Z = float(np.median(dvals))

    # 反投影（像素->相机系，单位 mm，X 右 Y 下 Z 前）
    X = (u - K["cx"]) * Z / K["fx"]
    Y = (v - K["cy"]) * Z / K["fy"]
    # 用 Z 与像素边长反算物理尺寸
    size_w = max(w, h) * Z / K["fx"]
    size_h = min(w, h) * Z / K["fy"]
    # minAreaRect 的角度归一到 [-45,45]
    yaw = ang if ang <= 45 else ang - 90

    vis = color.copy()
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
    cv2.putText(vis, f"Z={Z:.0f}mm yaw={yaw:.1f}", (int(u) - 60, int(v) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(f"{OUT}/{args.tag}_detect.png", vis)

    res = {
        "found": True,
        "touches_image_border": bool(best["touches"]),
        "warning": "轮廓贴到画面边缘/被遮挡，中心估计不可靠" if best["touches"] else "",
        "pixel": [round(u, 1), round(v, 1)],
        "depth_mm": round(Z, 1),
        "camera_xyz_mm": [round(X, 1), round(Y, 1), round(Z, 1)],
        "size_mm": [round(size_w, 1), round(size_h, 1)],
        "yaw_deg": round(yaw, 1),
        "area_px": int(area),
        "aspect": round(ar, 2),
        "solidity": round(sol, 2),
        "depth_valid_px": int(dvals.size),
        "intrinsics_source": K["source"],
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
