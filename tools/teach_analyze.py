#!/usr/bin/env python3
"""teach_analyze.py —— 把录下来的示教轨迹拆成"动作段 / 停顿点"，并算出关键位姿

用法: teach_analyze.py [轨迹文件] [--min-pause 0.6] [--min-move 3]
输出: 每个停顿点的 时间、关节角、法兰位姿，并把关键位姿存成 <轨迹>.keys.json
"""
import argparse, json, math, os, sys
import numpy as np

URDF = "/home/po/vision/agx_arm_urdf/piper/urdf/piper_description.urdf"
NAMES = ["J1","J2","J3","J4","J5","J6"]

_P = None
_B = None
_J = None

def _init():
    global _P, _B, _J
    if _P is not None:
        return
    import pybullet as p
    p.connect(p.DIRECT)
    _P = p
    _B = p.loadURDF(URDF, useFixedBase=True)
    _J = [j for j in range(p.getNumJoints(_B)) if p.getJointInfo(_B, j)[2] != p.JOINT_FIXED]

def fk(q_deg):
    _init()
    p = _P
    for j, v in zip(_J, q_deg):
        p.resetJointState(_B, j, math.radians(v))
    st = p.getLinkState(_B, 6)
    eul = p.getEulerFromQuaternion(st[5])
    return [v*1000 for v in st[4]], [math.degrees(v) for v in eul]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--min-pause", type=float, default=0.6)
    ap.add_argument("--min-move", type=float, default=3.0)
    a = ap.parse_args()
    path = a.path
    if path is None:
        d = "/home/po/vision/calib"
        cands = sorted((os.path.join(d,f) for f in os.listdir(d) if f.startswith("teach_") and f.endswith(".json")), key=os.path.getmtime)
        path = cands[-1]
    d = json.load(open(path))
    q = np.array(d["q"], float); t = np.array(d["t"], float)
    v = np.zeros(len(q))
    for i in range(1, len(q)):
        dt = max(1e-3, t[i]-t[i-1])
        v[i] = max(abs(q[i]-q[i-1]))/dt
    print("轨迹 %s  %.2f 秒  %d 点" % (os.path.basename(path), t[-1]-t[0], len(q)))

    # 分段: 速度>min_move 判"在动"，否则"停"
    segs = []
    state = v[0] > a.min_move
    t0 = 0
    for i in range(1, len(v)):
        st = v[i] > a.min_move
        if st != state:
            segs.append((state, t0, i-1))
            t0 = i; state = st
    segs.append((state, t0, len(v)-1))
    # 合并过短的停顿
    merged = []
    for seg in segs:
        if merged and (seg[1]-seg[0]) * 0.01 < a.min_pause:
            merged[-1] = (merged[-1][0], merged[-1][1], seg[2])
        else:
            merged.append(list(seg))
    print("分 %d 段：" % len(merged))
    keys = []
    for k, (moving, i0, i1) in enumerate(merged):
        dur = t[i1]-t[i0]
        qa, qb = q[i0], q[i1]
        dq = np.round(qb-qa, 1)
        pos0, rpy0 = fk(qa); pos1, rpy1 = fk(qb)
        if moving:
            print("  [%2d] %6.1f~%6.1f (%4.1fs) 动作  关节变化 %s" % (k, t[i0], t[i1], dur, dq.tolist()))
            print("        法兰 %.0f,%.0f,%.0f -> %.0f,%.0f,%.0f  rpy %.0f,%.0f,%.0f -> %.0f,%.0f,%.0f"
                  % (*pos0, *pos1, *rpy0, *rpy1))
        else:
            print("  [%2d] %6.1f~%6.1f (%4.1fs) 停顿  法兰 %.0f,%.0f,%.0f  rpy %.0f,%.0f,%.0f  q=%s"
                  % (k, t[i0], t[i1], dur, *pos1, *rpy1, np.round(qb,1).tolist()))
        keys.append({"seg": k, "kind": "move" if moving else "hold",
                     "t0": round(float(t[i0]),2), "t1": round(float(t[i1]),2),
                     "q": np.round(qb,2).tolist(), "flange": [round(v,1) for v in pos1],
                     "rpy": [round(v,1) for v in rpy1]})
    outp = path.replace(".json", ".keys.json")
    json.dump(keys, open(outp, "w"))
    print("关键位姿 -> %s" % outp)
    return 0

if __name__ == "__main__":
    sys.exit(main())
