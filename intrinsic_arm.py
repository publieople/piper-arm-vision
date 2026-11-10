#!/home/po/arm-env/bin/python
"""用机械臂运动做相机内参标定（板固定不动，相机扫视多个角度）

短板原因：原位姿全是近正面 ±10°，calibrateCamera 病态（fx/fy 差 9%、cy 偏 30px）。
这里用大角度 J4/J6 滚转 + J5 俯仰，让板在画面里呈现多种倾角。
输出: /home/po/vision/out/camera_intrinsics.npz  (fx fy cx cy K dist rms size)
数据: /home/po/vision/calib/intr_data.npz
"""
import math, os, time, json
import cv2, numpy as np
import handeye as he

CALIB, OUT = he.CALIB, he.OUT
PATTERN = he.PATTERN
SQUARE = he.SQUARE_MM / 1000.0
DATA = f"{CALIB}/intr_data.npz"

DELTAS = [
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 15, 0, 0), (0, 0, 0, -15, 0, 0),
    (0, 0, 0, 30, 0, 0), (0, 0, 0, -30, 0, 0),
    (0, 0, 0, 0, 0, 15), (0, 0, 0, 0, 0, -15),
    (0, 0, 0, 0, 0, 30), (0, 0, 0, 0, 0, -30),
    (0, 0, 0, 0, 12, 0), (0, 0, 0, 0, -12, 0),
    (0, 0, 0, 0, 20, 0), (0, 0, 0, 0, -20, 0),
    (0, 10, 0, 0, 0, 0), (0, -10, 0, 0, 0, 0),
    (8, 0, 0, 0, 0, 0), (-8, 0, 0, 0, 0, 0),
    (0, 6, 0, 18, 10, 18), (0, -6, 0, -18, -10, -18),
    (5, 5, 0, -15, 15, -12), (-5, -5, 0, 15, -15, 12),
]


def main():
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel="can0"))
    arm.connect(); time.sleep(1.0)
    base = he.joints_deg(arm)
    print("base joints:", " ".join("%.2f" % v for v in base))
    os.makedirs(f"{CALIB}/intr_imgs", exist_ok=True)
    corners = []
    ok_cnt = 0
    try:
        for i, d in enumerate(DELTAS):
            tgt = [b + x for b, x in zip(base, d)]
            arm.set_speed_percent(8)
            t_end = time.time() + 0.6
            while time.time() < t_end:
                arm.move_j([math.radians(v) for v in tgt]); time.sleep(0.02)
            time.sleep(0.7)
            tag = "in_%02d" % i
            try:
                he.capture(tag)
            except Exception as e:
                print("  pose %d capture fail %s" % (i, e)); continue
            img = cv2.imread(f"{OUT}/{tag}_color.png")
            cs = he.find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if cs is None:
                print("  pose %d: 没找到棋盘格，跳过 (%s)" % (i, " ".join("%.0f" % v for v in he.joints_deg(arm))))
                continue
            corners.append(cs); ok_cnt += 1
            print("  pose %d ok  joints %s" % (i, " ".join("%.1f" % v for v in he.joints_deg(arm))))
    finally:
        t_end = time.time() + 0.8
        while time.time() < t_end:
            arm.move_j([math.radians(v) for v in base]); time.sleep(0.02)
        arm.disconnect()

    np.savez(DATA, corners=np.array(corners))
    print("saved", DATA, "views", len(corners))
    if len(corners) < 6:
        raise SystemExit("有效视图太少")

    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    K0 = np.array([[480., 0, 320.], [0, 480., 240.], [0, 0, 1]])
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    rms, K, dist, rvs, tvs = cv2.calibrateCamera([objp] * len(corners), corners,
                                                (640, 480), K0.copy(), np.zeros(5), flags=flags)
    print("rms %.3f px  fx %.2f fy %.2f cx %.2f cy %.2f" % (rms, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("dist", np.round(dist.ravel(), 4))
    errs = []
    for i, (cs, rv, tv) in enumerate(zip(corners, rvs, tvs)):
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - cs.reshape(-1, 2), axis=1).mean()
        errs.append(e)
    print("per-view mean reproj err (px):", " ".join("%.2f" % e for e in errs))
    print("worst =", "%.2f" % max(errs))
    np.savez(f"{OUT}/camera_intrinsics.npz", fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
             K=K, dist=dist, rms=np.array(rms), size=np.array([640, 480]))
    print("saved", f"{OUT}/camera_intrinsics.npz")


if __name__ == "__main__":
    main()
