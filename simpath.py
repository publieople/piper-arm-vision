#!/usr/bin/env python3
"""simpath.py — 关节空间直线路径的仿真预检(桌面/自碰/限位/法兰最低点)

  /home/po/arm-env/bin/python simpath.py --q0 -11.1,82.7,-70.7,-0.7,48,1.2 --to 0,0,0,0,0,0
  不带 --q0 时用当前真机关节角。
输出: 全程最小桌面间隙、法兰最低 z、最紧关节余量; 不可行时打印原因。
"""
import os
import sys, math, argparse
import numpy as np
import pybullet as p

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agx_arm_urdf/piper/urdf/piper_with_gripper.urdf")
D = math.pi / 180.0
TABLE_Z = -0.075          # 桌面/垫子高度(基座系)
LIM = [(-150, 150), (0, 180), (-170, 0), (-100, 100), (-70, 70), (-120, 120)]


def build():
    p.connect(p.DIRECT)
    body = p.loadURDF(URDF, useFixedBase=True)
    joints = [j for j in range(p.getNumJoints(body)) if p.getJointInfo(body, j)[2] == p.JOINT_REVOLUTE]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.5, 1.5, 0.01])
    table = p.createMultiBody(0, col, basePosition=[0.25, 0, TABLE_Z - 0.01])
    return body, joints, table


def set_q(body, joints, q):
    for j, a in zip(joints, q):
        p.resetJointState(body, j, a * D)


def min_gap(body, other, skip=("base_link", "link1")):
    best, link = 999.0, None
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode()
        if name in skip:
            continue
        for pt in p.getClosestPoints(body, other, 0.40, linkIndexA=j):
            if pt[8] * 1000 < best:
                best, link = pt[8] * 1000, name
    return best, link


def lowest_z(body):
    """末端几个 link(法兰/夹爪) 的最低点 z(mm) —— 决定会不会压到桌面"""
    lo, link = 999.0, None
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode()
        if not any(k in name for k in ("link5", "link6", "flange", "gripper")):
            continue
        ls = p.getLinkState(body, j, computeForwardKinematics=True)
        z = ls[4][2] * 1000
        if z < lo:
            lo, link = z, name
    return lo, link


def check_path(q_from, q_to, n=36, verbose=True):
    body, joints, table = build()
    res = {"ok": True, "min_table": 999.0, "min_table_at": None, "min_flange_z": 999.0,
           "min_z_link": None, "bad_limits": [], "path": []}
    for k in range(n + 1):
        q = [a + (b - a) * k / n for a, b in zip(q_from, q_to)]
        set_q(body, joints, q)
        gap, link = min_gap(body, table)
        lz, lzl = lowest_z(body)
        bad = [i + 1 for i, (a, (lo, hi)) in enumerate(zip(q, LIM)) if not (lo - 0.5 <= a <= hi + 0.5)]
        if bad:
            res["bad_limits"].append((round(k / n, 2), bad))
            res["ok"] = False
        if gap < res["min_table"]:
            res["min_table"], res["min_table_at"] = gap, (round(k / n, 2), link)
        if lz < res["min_flange_z"]:
            res["min_flange_z"], res["min_z_link"] = lz, lzl
        res["path"].append((round(k / n, 2), round(gap, 1), round(lz, 1)))
    if res["min_table"] < 10.0:
        res["ok"] = False
    if verbose:
        print("[simpath] 桌面最小间隙 %.1f mm @ %s   最低点 z=%.1f mm (%s)   越限 %s"
              % (res["min_table"], res["min_table_at"], res["min_flange_z"], res["min_z_link"],
                 res["bad_limits"] or "无"))
        for k, gap, lz in res["path"][::6]:
            print("   %.2f  桌面间隙 %6.1f mm  最低 z %7.1f mm" % (k, gap, lz))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q0", dest="frm", default=None, help="起始关节角, 逗号分隔(可负数)")
    ap.add_argument("--to", dest="to", default="0,0,0,0,0,0")
    ap.add_argument("--n", type=int, default=36)
    a = ap.parse_args()
    if a.frm:
        q0 = [float(v) for v in a.frm.split(",")]
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import autograsp as A
        arm, _ = A.connect()
        q0 = A.cur_joints()
        arm.disconnect()
        print("[simpath] 当前关节", q0)
    qt = [float(v) for v in a.to.split(",")]
    r = check_path(q0, qt, n=a.n)
    print("[simpath]", "可行" if r["ok"] else "不可行")


if __name__ == "__main__":
    main()
