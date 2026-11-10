#!/usr/bin/env python3
"""kin.py — PiPER 运动学预检(纯仿真, 不连真机)

用途: 任何 move_p 之前先问一句"这个位姿解得出吗、哪个关节没余量",
      避免真机发出去被控制器静默忽略(踩过多次)。

  import kin
  info = kin.check([357, -50, 229], [180, 0, 138.7])
  # info = {ok, q, err, margins:{J1..J6}, worst, msg}

模型: 官方 URDF piper_description.urdf, 与真机 FK 实测差 0.06mm。
关节限位(度): J1 ±150, J2 -178..180, J3 -170..0, J4 ±100, J5 ±70, J6 ±115
"""
import math
import os

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agx_arm_urdf/piper/urdf/piper_description.urdf")
LO = [-150.0, -178.0, -170.0, -100.0, -70.0, -115.0]
HI = [150.0, 180.0, 0.0, 100.0, 70.0, 115.0]
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
FLANGE_LINK = 6
SEEDS = 48

_p = None
_body = None
_joints = None


def _lazy():
    """第一次调用才 import pybullet 并加载模型(约 0.4s)"""
    global _p, _body, _joints
    if _p is not None:
        return True
    try:
        import pybullet as p
    except Exception as e:                                   # 没装 pybullet 时降级
        print("[kin] 略过预检: %s" % e)
        return False
    _p = p
    p.connect(p.DIRECT)
    _body = p.loadURDF(URDF, useFixedBase=True)
    _joints = [j for j in range(p.getNumJoints(_body))
               if p.getJointInfo(_body, j)[2] != p.JOINT_FIXED]
    for j, lo, hi in zip(_joints, LO, HI):
        p.changeDynamics(_body, j, jointLowerLimit=math.radians(lo),
                         jointUpperLimit=math.radians(hi), jointLimitForce=1e9)
    return True


def _solve(xyz_mm, rpy_deg, seed=None):
    p = _p
    Q = p.getQuaternionFromEuler([math.radians(v) for v in rpy_deg])
    seed = seed or [0, 90, -60, 0, 50, 0]
    q = p.calculateInverseKinematics(
        _body, FLANGE_LINK, [v / 1000.0 for v in xyz_mm], Q,
        lowerLimits=[math.radians(v) for v in LO],
        upperLimits=[math.radians(v) for v in HI],
        jointRanges=[math.radians(HI[i] - LO[i]) for i in range(6)],
        restPoses=[math.radians(v) for v in seed],
        maxNumIterations=400, residualThreshold=1e-8)
    qs = [math.degrees(v) for v in q[:6]]
    for j, v in zip(_joints, q[:6]):
        p.resetJointState(_body, j, v)
    st = p.getLinkState(_body, FLANGE_LINK)
    err = math.dist(st[4], [v / 1000.0 for v in xyz_mm]) * 1000.0
    return err, [round(v, 1) for v in qs]


def solve(xyz_mm, rpy_deg, seed=None, tries=SEEDS, err_tol=3.0):
    """多种子求逆解; 返回 (err_mm, q_deg) 最好的一组, 无解返回 (inf, None)"""
    if not _lazy():
        return None, None
    import random
    import numpy as np
    best = (float("inf"), None)
    seeds = [seed] if seed else [None]
    rnd = random.Random(0)
    seeds += [[rnd.uniform(LO[k] + 3, HI[k] - 3) for k in range(6)] for _ in range(tries)]
    for sd in seeds:
        err, q = _solve(xyz_mm, rpy_deg, sd)
        ok = all(LO[k] + 0.2 <= q[k] <= HI[k] - 0.2 for k in range(6))
        if err < err_tol and ok and err < best[0]:
            best = (err, q)
    return best


def margins(q):
    """每个关节到最近限位的余量(度), 越小越危险"""
    if q is None:
        return None
    return {NAMES[k]: round(min(q[k] - LO[k], HI[k] - q[k]), 1) for k in range(6)}


def check(xyz_mm, rpy_deg, seed=None, min_margin=2.0):
    """可达性预检

    ok=False 一定不可达(多组种子都解不出);
    warn=True 表示解得出但某关节余量 < min_margin, 真机可能被静默拒绝。
    """
    if not _lazy():
        return dict(ok=True, warn=False, q=None, err=None, margins=None,
                    worst=None, msg="未做预检(pybullet 不可用)")
    err, q = solve(xyz_mm, rpy_deg, seed)
    if q is None:
        return dict(ok=False, warn=False, q=None, err=err, margins=None, worst=None,
                    msg="IK 无解: 该位姿超出关节限位/臂展")
    m = margins(q)
    worst = min(m.values())
    return dict(ok=True, warn=worst < min_margin, q=q, err=round(err, 2), margins=m,
                worst=worst,
                msg="可达 err=%.1fmm 最紧关节余量 %.1f deg" % (err, worst))


if __name__ == "__main__":
    import sys
    cases = [([357, -50, 229], [180, 0, 138.7], "上次卡死的高位"),
             ([280, -84, 237], [180, 0, -134], "实际悬停位"),
             ([281, -84, 156], [180, 0, -134], "实际抓取位"),
             ([60, 0, 213], [0, 85, 0], "home 位")]
    if len(sys.argv) > 6:
        cases = [([float(v) for v in sys.argv[1:4]], [float(v) for v in sys.argv[4:7]], "命令行")] 
    for xyz, rpy, name in cases:
        r = check(xyz, rpy)
        print("%-14s %s -> ok=%s %s" % (name, xyz, r["ok"], r["msg"]))
