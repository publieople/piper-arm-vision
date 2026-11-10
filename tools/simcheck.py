#!/usr/bin/env python3
"""PiPER 运动学预检（PyBullet + 官方 URDF）——无真机动作。

用法:
  simcheck.py fk <j1..j6 度>
  simcheck.py compare            # 对比当前真机位姿与 URDF FK
  simcheck.py ik <x> <y> <z> <roll> <pitch> <yaw>   # 目标(米/度) -> 关节解 + 关节跳变
"""
import sys, math, json, os
import pybullet as p
import numpy as np

URDF = "/home/po/vision/agx_arm_urdf/piper/urdf/piper_description.urdf"
D = math.pi / 180.0

def load():
    cid = p.connect(p.DIRECT)
    body = p.loadURDF(URDF, useFixedBase=True)
    joints, links = [], {}
    for j in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, j)
        if info[2] == p.JOINT_REVOLUTE:
            joints.append(j)
        links[info[12].decode()] = j
    return cid, body, joints, links

def fk(body, joints, q_deg):
    for j, a in zip(joints, q_deg):
        p.resetJointState(body, j, a * D)
    ls = p.getLinkState(body, joints[-1] + 0, computeForwardKinematics=True)
    # 用 joint6 的子 link：link6
    return ls

def link_of(links, name):
    return links[name]

def pose_of(body, link):
    ls = p.getLinkState(body, link, computeForwardKinematics=True)
    pos = np.array(ls[4]); quat = np.array(ls[5])
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return pos, R

def R_to_euler_xyz_extrinsic(R):
    # 与 SDK 一致: R = Rz(y) Ry(p) Rx(r)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return roll, pitch, yaw

def euler_to_R(r, pf, y):
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(pf),0,math.sin(pf)],[0,1,0],[-math.sin(pf),0,math.cos(pf)]])
    Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def main():
    cid, body, joints, links = load()
    flange = link_of(links, "link6")
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"

    if mode == "fk":
        q = [float(v) for v in sys.argv[2:8]]
        pos, R = pose_of(body, flange)
        r, pf, y = R_to_euler_xyz_extrinsic(R)
        print("FK pos mm", np.round(pos*1000, 2).tolist())
        print("FK rpy deg", np.round([r/D, pf/D, y/D], 2).tolist())
        return

    if mode == "compare":
        q = [float(v) for v in sys.argv[2:8]]
        for j, a in zip(joints, q):
            p.resetJointState(body, j, a * D)
        pos, R = pose_of(body, flange)
        r, pf, y = R_to_euler_xyz_extrinsic(R)
        print(json.dumps({
            "urdf_pos_mm": np.round(pos*1000, 2).tolist(),
            "urdf_rpy_deg": np.round([r/D, pf/D, y/D], 2).tolist(),
        }))
        return

    if mode == "ik":
        x, y, z, rr, pp, yy = [float(v) for v in sys.argv[2:8]]
        q_now = [float(v) for v in sys.argv[8:14]] if len(sys.argv) >= 14 else [0]*6
        target_pos = [x, y, z]
        target_orn = list(p.getQuaternionFromEuler([rr*D, pp*D, yy*D]))
        lower = [-2.618, 0, -2.967, -1.745, -1.2217, -2.967]
        upper = [2.618, 3.1416, 0, 1.745, 1.2217, 2.967]
        rest = [a*D for a in q_now]
        sol = p.calculateInverseKinematics(body, flange, target_pos, target_orn,
                                          lower, upper, [u-l for l,u in zip(lower,upper)],
                                          rest, maxNumIterations=500, residualThreshold=1e-6)
        sol6 = [s/D for s in sol[:6]]
        delta = [a-b for a, b in zip(sol6, q_now)]
        # 用解算结果做 FK 验证
        for j, a in zip(joints, sol[:6]):
            p.resetJointState(body, j, a)
        pos, R = pose_of(body, flange)
        err = np.linalg.norm(pos - np.array(target_pos))*1000
        print(json.dumps({
            "ik_joint_deg": np.round(sol6, 2).tolist(),
            "delta_deg": np.round(delta, 2).tolist(),
            "max_delta": float(np.max(np.abs(delta))),
            "fk_err_mm": round(float(err), 3),
            "within_limits": bool(all(l-1e-6 <= s <= u+1e-6 for s, l, u in zip(sol[:6], lower, upper))),
        }))
        return

if __name__ == "__main__":
    main()
