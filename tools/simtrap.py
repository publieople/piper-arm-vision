#!/usr/bin/env python3
"""整条抓取轨迹的碰撞/限位仿真预演（PyBullet + 官方 URDF）。

阶段:
  0 当前位姿 -> 1 中间位 -> 2 预备位(IK解) -> 3 直线下降 80mm -> 4 抓取高度
输出: 每阶段关节角、末端位姿、与桌面/盒子的最小间隙。
"""
import math, json, sys
import pybullet as p
import numpy as np

URDF = "/home/po/vision/agx_arm_urdf/piper/urdf/piper_description.urdf"
D = math.pi / 180.0

CUR = [-3.81, 2.97, -2.87, -4.56, 29.99, 1.79]
PRE = [-15.50, 126.52, -68.29, 0.07, 36.75, 77.42]
BOX = (0.3576, -0.098)
BOX_TOP_Z = -0.0335
BOX_SIZE = (0.106, 0.062, 0.030)   # 长 宽 高(m) 估计
TABLE_Z = -0.063                  # 桌面(基座系)
APPROACH = 0.080

limits_low = [-2.618, 0.0, -2.967, -1.745, -1.2217, -2.967]
limits_up = [2.618, 3.1416, 0.0, 1.745, 1.2217, 2.967]


def build():
    p.connect(p.DIRECT)
    body = p.loadURDF(URDF, useFixedBase=True)
    joints = []
    for j in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, j)
        if info[2] == p.JOINT_REVOLUTE:
            joints.append(j)
    # 桌面
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.2, 1.2, 0.01])
    table_id = p.createMultiBody(0, table_col, basePosition=[0.3, 0, TABLE_Z - 0.01])
    # 盒子
    box_col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[BOX_SIZE[0]/2, BOX_SIZE[1]/2, BOX_SIZE[2]/2])
    box_id = p.createMultiBody(0, box_col,
                               basePosition=[BOX[0], BOX[1], BOX_TOP_Z - BOX_SIZE[2]/2])
    return body, joints, table_id, box_id


def set_q(body, joints, q_deg):
    for j, a in zip(joints, q_deg):
        p.resetJointState(body, j, a * D)


def clearance(body, other, skip=("base_link", "link1")):
    best = 999.0
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode()
        if name in skip:
            continue
        pts = p.getClosestPoints(body, other, 0.12, linkIndexA=j)
        for pt in pts:
            best = min(best, pt[8] * 1000)
    return best


def flange_pose(body, link):
    ls = p.getLinkState(body, link, computeForwardKinematics=True)
    return np.array(ls[4]), np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)


def check(q, body, joints, link, table_id, box_id, label):
    set_q(body, joints, q)
    pos, R = flange_pose(body, link)
    bad = [i + 1 for i, (a, lo, hi) in enumerate(zip(q, limits_low, limits_up))
           if not (lo/D - 1e-6 <= a <= hi/D + 1e-6)]
    c_tab = clearance(body, table_id)
    c_box = clearance(body, box_id)
    print("%-10s q=%s  flange=%s mm  间隙: 桌 %.1f mm 盒 %.1f mm  越限:%s"
          % (label, np.round(q, 1).tolist(), np.round(pos*1000, 1).tolist(), c_tab, c_box, bad or "无"))
    return c_tab, c_box


def main():
    body, joints, table_id, box_id = build()
    link = joints[-1] + 0  # joint6 -> 其子 link 需要查
    # 找 link6 索引
    link = None
    for j in range(p.getNumJoints(body)):
        if p.getJointInfo(body, j)[12].decode() == "link6":
            link = j
    if link is None:
        link = p.getNumJoints(body) - 1

    check(CUR, body, joints, link, table_id, box_id, "0 当前")
    mid = [(a+b)/2 for a, b in zip(CUR, PRE)]
    check(mid, body, joints, link, table_id, box_id, "1 中间位")
    check(PRE, body, joints, link, table_id, box_id, "2 预备位")

    # 直线下降: 从 box_top+APPROACH 到 box_top+0.005
    z0 = BOX_TOP_Z + APPROACH
    z1 = BOX_TOP_Z + 0.005
    lower = limits_low
    upper = limits_up
    orn = p.getQuaternionFromEuler([math.pi, 0, 87 * D])
    min_tab, min_box = 999, 999
    worst = None
    for k in range(9):
        z = z0 + (z1 - z0) * k / 8.0
        sol = p.calculateInverseKinematics(body, link, [BOX[0], BOX[1], z], orn,
                                           lower, upper,
                                           [u - l for l, u in zip(lower, upper)],
                                           [a * D for a in PRE], maxNumIterations=400,
                                           residualThreshold=1e-8)
        q = [s / D for s in sol[:6]]
        set_q(body, joints, q)
        pos, R = flange_pose(body, link)
        err = np.linalg.norm(pos - np.array([BOX[0], BOX[1], z])) * 1000
        ct, cb = clearance(body, table_id), clearance(body, box_id)
        if k in (0, 4, 8):
            print("  下降 z=%.1f mm  q=%s  末端=%.1f,%.1f,%.1f  err=%.1f  桌%.1f 盒%.1f"
                  % (z*1000, np.round(q, 1).tolist(), pos[0]*1000, pos[1]*1000, pos[2]*1000,
                     err, ct, cb))
        if ct < min_tab:
            min_tab, worst = ct, ("桌", k, q)
        min_box = min(min_box, cb)
    print("直线下降最小间隙: 桌面 %.1f mm / 盒子 %.1f mm" % (min_tab, min_box))

    # 全轨迹关节空间插值检查
    allq = []
    for seg in [(CUR, mid), (mid, PRE)]:
        for k in range(11):
            allq.append([a + (b - a) * k / 10.0 for a, b in zip(*seg)])
    lim_viol = [(i, round(q[i], 1)) for q in allq for i in range(6)
                if not (limits_low[i]/D - 1e-6 <= q[i] <= limits_up[i]/D + 1e-6)]
    print("关节空间段越限点:", lim_viol or "无")
    maxstep = max(max(abs(a - b) for a, b in zip(allq[k], allq[k+1])) for k in range(len(allq)-1))
    print("相邻插值最大单关节步长: %.1f 度" % maxstep)


if __name__ == "__main__":
    main()
