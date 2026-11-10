#!/usr/bin/env python3
"""bottlegrasp.py -- 斜夹 35 度抓饮料瓶(空瓶)
位姿: 工具轴前倾 35 度, 从瓶子上方前侧压下去, 指板贴瓶身直壁;
法兰 z = 瓶顶 - 34 + 87, 沿工具轴后退 120mm 为预备位.
"""
import os
import sys, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import pybullet as PB
import autograsp as A

KD = np.load("/home/po/vision/out/camera_intrinsics.npz"); K = KD["K"]; DIST = KD["dist"]
Xd = np.load("/home/po/vision/out/hand_eye.npz")
X = np.eye(4); X[:3,:3] = Xd["R"]; X[:3,3] = Xd["t"].ravel()

URDF = "/home/po/vision/agx_arm_urdf/piper/urdf/piper_description.urdf"
LO = [-150,-178,-170,-100,-70,-115]; HI = [150,180,0,100,70,115]
_cid = PB.connect(PB.DIRECT); _rob = PB.loadURDF(URDF, useFixedBase=True)
_JS = [j for j in range(PB.getNumJoints(_rob)) if PB.getJointInfo(_rob,j)[2] != PB.JOINT_FIXED]
for _j,_lo,_hi in zip(_JS,LO,HI):
    PB.changeDynamics(_rob,_j,jointLowerLimit=math.radians(_lo),jointUpperLimit=math.radians(_hi),jointLimitForce=1e9)

def ik(xyz, rpy, seed):
    q = PB.calculateInverseKinematics(_rob, 6, [v/1000.0 for v in xyz],
            PB.getQuaternionFromEuler([math.radians(v) for v in rpy]),
            lowerLimits=[math.radians(v) for v in LO], upperLimits=[math.radians(v) for v in HI],
            jointRanges=[math.radians(b-a) for a,b in zip(LO,HI)],
            restPoses=[math.radians(v) for v in seed],
            maxNumIterations=500, residualThreshold=1e-8)
    for j,qq in zip(_JS,q): PB.resetJointState(_rob,j,qq)
    st = PB.getLinkState(_rob,6,computeForwardKinematics=True)
    err = float(np.linalg.norm(np.array(st[4])*1000-np.array(xyz)))
    qd = [round(math.degrees(v),1) for v in q]
    bad = [i+1 for i,v in enumerate(qd) if v < LO[i]-0.5 or v > HI[i]+0.5]
    return qd, err, bad

def e2R(r,p_,y_):
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry = np.array([[math.cos(p_),0,math.sin(p_)],[0,1,0],[-math.sin(p_),0,math.cos(p_)]])
    Rz = np.array([[math.cos(y_),-math.sin(y_),0],[math.sin(y_),math.cos(y_),0],[0,0,1]])
    return Rz@Ry@Rx

def flange_T(arm):
    p, rpy = A.pose(arm)
    T = np.eye(4); T[:3,:3] = e2R(*np.radians(rpy)); T[:3,3] = p/1000.0
    return T, p, rpy

# ---------------------------------------------------------------- 检测
def find_bottle(bgr, dep, tag="bot"):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    best = None
    # 暖色(橙/黄) + 红色(hue 在 160-180 也有一段, 2026-11-08 实测红瓶正落在这里)
    for lo,hi in (((5,80,60),(40,255,255)), ((5,45,40),(40,255,255)), ((0,30,30),(45,255,255)),
                  ((160,70,50),(180,255,255)), ((165,50,40),(180,255,255))):
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
        c = None
        for cand in cnts:
            if cv2.contourArea(cand) < 1200: continue
            m3 = np.zeros(mask.shape, np.uint8); cv2.drawContours(m3,[cand],-1,255,-1)
            ys3,xs3 = np.where(m3>0)
            dv3 = dep[ys3, xs3]; dv3 = dv3[dv3>0]
            if dv3.size < 0.3*len(xs3): continue          # 深度基本无效 = 远处背景(椅子/墙)
            if float(np.median(dv3)) > 700: continue      # 太远, 不是桌上的瓶子
            c = cand; break
        if c is None: continue
        m2 = np.zeros(mask.shape, np.uint8); cv2.drawContours(m2,[c],-1,255,-1)
        ys,xs = np.where(m2>0)
        top = int(ys.min()); bot = int(ys.max())
        band = max(10, int(0.10*(bot-top)))
        sel = ys < top + band
        u = float(np.median(xs[sel])); v = float(top + 3)
        dv = dep[ys[sel], xs[sel]]; dv = dv[dv>0]
        dmed = float(np.median(dv)) if dv.size else -1.0
        best = {"u":u,"v":v,"depth":dmed,"top":top,"bot":bot,"w":float(xs.max()-xs.min()),
                "h":float(bot-top),"area":float(cv2.contourArea(c))}
        cv2.rectangle(bgr,(int(xs.min()),top),(int(xs.max()),bot),(0,255,0),2)
        cv2.circle(bgr,(int(u),int(v)),5,(0,0,255),-1)
        cv2.imwrite(f"/home/po/vision/out/{tag}_detect.png", bgr)
        if dmed > 0: break
    return best

DS = 1.43   # 深度比例修正: 实测深度比真实距离小 1.43 倍(像素/内参与深度不自洽), main 里 --ds 可覆盖

def bottle_base(arm, tag="bot"):
    bgr, dep = A.capture(tag)
    d = find_bottle(bgr, dep, tag)
    if d is None: return None, "看不见瓶子(暖色块没找到)"
    zr = d["depth"]
    z = zr * DS
    if zr <= 0: return None, f"瓶盖深度无效({zr:.0f})"
    if zr < 150 or zr > 600: return None, f"瓶盖深度异常(原始) {zr:.0f}mm"
    und = cv2.undistortPoints(np.array([[[d["u"],d["v"]]]],np.float64),K,DIST,P=K).reshape(2)
    pc = np.array([(und[0]-K[0,2])*z/K[0,0], (und[1]-K[1,2])*z/K[1,1], z])/1000.0
    T,p,rpy = flange_T(arm)
    pb = (T @ X @ np.append(pc,1.0))[:3]*1000.0
    return pb, f"px=({d['u']:.0f},{d['v']:.0f}) 盖深 {zr:.0f}->{z:.0f}mm(x{DS:.2f}) 块 {d['w']:.0f}x{d['h']:.0f}px (高{d['h']:.0f}px)"

def plan(pb, tilt=35.0, pad=106.5, rad=30.0):
    """pb = 瓶身实测点(表面, 不是轴线) -> (法兰, 预备, rpy, d)
    rad = 瓶半径: 实测点在圆柱正面, 轴线还要沿径向外推 rad"""
    az = math.atan2(pb[1], pb[0])
    contact = np.array([pb[0] + rad*math.cos(az), pb[1] + rad*math.sin(az), pb[2]-34.0])
    d = np.array([math.sin(math.radians(tilt))*math.cos(az),
                  math.sin(math.radians(tilt))*math.sin(az),
                  -math.cos(math.radians(tilt))])
    fl = contact - pad*d
    pre = fl - 120.0*d
    rpy = (180.0, tilt, 180.0 + math.degrees(az))
    return fl, pre, rpy, d

def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", action="store_true", help="夹住后抬 50mm(默认只夹不抬)")
    ap.add_argument("--dry", action="store_true", help="只算不动")
    ap.add_argument("--tilt", type=float, default=35.0)
    ap.add_argument("--ds", type=float, default=1.0, help="深度比例修正(实测 1.0 与桌面基准吻合)")
    ap.add_argument("--r", type=float, default=30.0, help="瓶半径 mm")
    a = ap.parse_args()
    DS = a.ds

    arm, eff = A.connect()
    st = arm.get_arm_status().msg
    print(f"[arm] ctrl={st.ctrl_mode} arm={st.arm_status} motion={st.motion_status}")
    q_now = A.cur_joints()
    print(f"[arm] 现在关节 {q_now}")
    A.gripper(eff, A.CFG["open_mm"], 1.0)

    # 1) 测量位(该位姿瓶子整只入画)
    print("\n--- 测量 ---")
    A.goto_q(arm, [0,0,0,0,20,0], steps=3, speed=10)
    p, rpy = A.pose(arm)
    print(f"[meas] 法兰 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) rpy ({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})")
    pb = None
    for i in range(3):
        pb, info = bottle_base(arm, f"bv{i}")
        if pb is not None: break
        print(f"[meas] 第{i+1}次没测到: {info}")
        time.sleep(0.3)
    if pb is None: sys.exit(f"测不到瓶子: {info}")
    r = math.hypot(pb[0], pb[1])
    print(f"[bottle] 顶 ({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f}) mm  半径 {r:.0f}mm  | {info}")

    # 2) 规划
    fl, pre, grpy, d = plan(pb, a.tilt, rad=a.r)
    print(f"\n--- 规划 ---")
    print(f"  工具轴 d=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f})  前倾 {a.tilt:.0f} 度")
    print(f"  夹取法兰 ({fl[0]:.0f},{fl[1]:.0f},{fl[2]:.0f}) rpy ({grpy[0]:.0f},{grpy[1]:.0f},{grpy[2]:.0f})")
    print(f"  预备法兰 ({pre[0]:.0f},{pre[1]:.0f},{pre[2]:.0f}) (后退 120mm)")
    q_pre, e1, b1 = ik(pre, grpy, q_now)
    q_grp, e2, b2 = ik(fl, grpy, q_pre)
    print(f"  IK闭预备 err={e1:.1f}mm {'越限J'+str(b1) if b1 else 'ok'} q={q_pre}")
    print(f"  IK闭夹取 err={e2:.1f}mm {'越限J'+str(b2) if b2 else 'ok'} q={q_grp}")
    if e1 > 6 or e2 > 6 or b1 or b2:
        sys.exit("IK 不可行, 不动")
    if a.dry:
        print("[dry] 到此为止"); return

    # 3) 关节空间到预备位
    print("\n--- 走到预备位 ---")
    A.goto_q(arm, q_pre, steps=5, speed=10)
    p, rpy = A.pose(arm)
    print(f"[pre] 法兰 ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) 残差 {np.linalg.norm(p-pre):.1f}mm")

    # 4) 复测(近距离更准)
    pb2, info2 = bottle_base(arm, "bv_pre")
    if pb2 is not None and np.linalg.norm(np.array(pb2[:2])-np.array(pb[:2])) < 40:
        print(f"[pre] 复测 顶 ({pb2[0]:.0f},{pb2[1]:.0f},{pb2[2]:.0f}) 与远处差 {np.linalg.norm(np.array(pb2)-np.array(pb)):.0f}mm 用复测")
        pb = pb2
        fl, pre2, grpy, d = plan(pb, a.tilt, rad=a.r)
        q_grp, e2, b2 = ik(fl, grpy, q_grp)
        print(f"  重算夹取 IK err={e2:.1f}mm q={q_grp}")
    else:
        print(f"[pre] 复测没结果({info2}), 沿用远处测量")

    # 5) 沿工具轴分段直线下去
    print("\n--- 直线进给 ---")
    p0, _ = A.pose(arm)
    dist = float(np.linalg.norm(fl - p0))
    n = max(3, int(math.ceil(dist/20.0)))
    ok_all = True
    for i in range(1, n+1):
        tgt = p0 + (fl - p0)*i/n
        ok, err = A.move_p(arm, [*tgt, *grpy], speed=6, timeout=10)
        p2, _ = A.pose(arm)
        step_err = float(np.linalg.norm(p2-tgt))
        print(f"  {i}/{n} 目标 ({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}) 到 ({p2[0]:.0f},{p2[1]:.0f},{p2[2]:.0f}) 偏 {step_err:.1f}mm")
        if step_err > 8.0:
            ok_all = False; A.soft_stop(arm); print("  偏差过大, 停"); break
    if not ok_all: sys.exit(2)
    pf, _ = A.pose(arm)
    print(f"[grasp] 到位 ({pf[0]:.0f},{pf[1]:.0f},{pf[2]:.0f}) 距目标 {np.linalg.norm(pf-fl):.1f}mm")

    # 6) 闭合
    print("\n--- 闭合 ---")
    w0, f0 = A.gripper(eff, 45.0, 2.5)
    print(f"[grip] 宽={w0}mm 力={f0}N")
    hit = (w0 > 48.0 and abs(f0) > 0.8)
    print("[grip] " + ("夹到东西了" if hit else "夹空了(宽度缩到 45 说明没碰到瓶子)"))
    if not hit:
        A.gripper(eff, A.CFG["open_mm"], 1.0); sys.exit(3)

    # 7) 抬
    if a.lift:
        print("\n--- 抬 50mm ---")
        p1, _ = A.pose(arm)
        up = p1.copy(); up[2] += 50.0
        A.move_p(arm, [*up, *grpy], speed=6, timeout=10)
        w1, f1 = A.gripper(eff, 45.0, 2.5, seconds=0.5)
        p2, _ = A.pose(arm)
        print(f"[lift] 抬到 z={p2[2]:.0f} 宽={w1}mm 力={f1}N -> " + ("抓住了" if abs(f1) > 0.8 else "掉了"))
    print("\n[done]")

if __name__ == "__main__":
    main()
