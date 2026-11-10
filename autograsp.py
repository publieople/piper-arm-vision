#!/usr/bin/env python3
"""autograsp.py -- 桌面方盒全自动抓取（纯代码，不依赖任何 LLM / 神经网络）

环境: /home/po/arm-env/bin/python  (pyAgxArm + python-can + numpy + opencv)
取图: 调 /home/po/vision-env/bin/python capture.py (Orbbec SDK 只有 cp311 wheel)

用法:
  autograsp.py --calib        # 一次: 测像素->基座的伺服雅可比, 存 calib/servo_calib.json
  autograsp.py                # 抓一次: 观察->对中->下降->闭合->抬起
  autograsp.py --release      # 放回: 降回抓取高度->张开->抬走

示教常量(已在本机实测, 换环境要重测):
  obs_z=189      观察/对中高度(法兰 z, mm)
  grasp_z=139    抓取高度(法兰 z, mm)  —— 指尖落在盒子上半部
  target_px      观察高度下盒子正对夹爪时的质心像素
"""
import argparse, atexit, json, math, os, subprocess, sys, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armraw
import kin

VISION_PY = "/home/po/vision-env/bin/python"
CAPTURE   = "/home/po/vision/capture.py"
OUT       = "/home/po/vision/out"
CALIB     = "/home/po/vision/calib/servo_calib.json"

CFG = {
    "obs_z": 228.0,
    "obs_xy": (284.0, -60.0),
    "obs_rpy": (180.0, 0.0, 168.0),     # deg (J6=0 时的法兰滚转, 让腕子两侧都有余量)
    "target_px": (382.2, 281.5),        # 盒子质心应落在的像素
    "grasp_z": 139.0,
    "open_mm": 78.0,
    "close_mm": 58.0,
    "close_n": 3.0,
    "lift_mm": 50.0,
    "speed": 5,
    "step_mm": 10.0,
    "depth_min": 80,
    "depth_max": 400,
    "max_iter": 8,
    "max_lateral_mm": 15.0,
    # 角度伺服: 目标 = 矩形边与画面轴对齐 (误差取 mod 90), 增益由 --calib 测
    "yaw_tol": 3.0,
    "yaw_gain": 1.0,
    "yaw_step": 15.0,
    # 示教关节位姿: 大范围移动一律走 move_j 关节空间(笛卡尔长距离会撞奇异区)
    "home_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "obs_q":  [-11.86, 92.52, -62.95, 0.0, 65.57, 0.0],
}

_ARM = None

# ---------------------------------------------------------------- arm
def connect():
    from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel="can0"))
    arm.connect(); time.sleep(1.2)
    eff = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER); time.sleep(0.8)
    global _ARM
    _ARM = arm
    return arm, eff

def pose(arm):
    f = arm.get_flange_pose().msg
    return np.array([f[0]*1000, f[1]*1000, f[2]*1000]), np.degrees(np.array(f[3:]))

# ---------------------------------------------------------------- 安全前置
IK_CHECK = True          # move_p 前做可达性预检(kin.py); 置 False 可关
REC_PATH = None          # 非 None 时把每步写 JSONL


def ensure_ctrl(arm):
    """动作前确保在 CAN_CTRL。用户手拖过机械臂后会留在 TEACHING_MODE,
    此状态下所有 move_j/move_p 都被忽略(且照报"已到达"), 踩过一次脱力事故。"""
    try:
        st = str(arm.get_arm_status().msg.ctrl_mode)
    except Exception:
        return True
    if "TEACHING" not in st:
        return True
    print("[teach] 当前处于示教模式, 复位到 CAN_CTRL ...")
    arm.enable(joint_index=255)
    time.sleep(0.3)
    q = cur_joints() or [0.0] * 6
    q = list(q)
    q[4] += 2.0                          # 小幅度 nudged, 只为触发模式切换
    arm.set_speed_percent(5)
    t0 = time.time()
    while time.time() - t0 < 1.2:
        arm.move_j([math.radians(v) for v in q])
        time.sleep(0.02)
    time.sleep(0.5)
    try:
        st2 = str(arm.get_arm_status().msg.ctrl_mode)
    except Exception:
        return True
    if "TEACHING" in st2:
        print("[teach] 仍在示教模式: 需要人手按一下示教按钮")
        return False
    print("[teach] 已回到 CAN_CTRL")
    return True


def ik_precheck(xyzrpy, force=False):
    """可达性预检: 解不出就别发(真机会静默忽略, 白等 8 秒还以为是到位了)"""
    if not IK_CHECK:
        return True
    info = kin.check(xyzrpy[:3], xyzrpy[3:], seed=cur_joints())
    if not info["ok"]:
        print("[ik] %s  %s" % (np.round(np.asarray(xyzrpy[:3]), 0).tolist(), info["msg"]))
        return bool(force)
    if info["warn"]:
        m = info["margins"]
        tight = min(m, key=lambda k: m[k])
        print("[ik] 余量小: %s=%.1f deg (err %.1fmm)" % (tight, m[tight], info["err"]))
    return True


def record(event, **kw):
    """把每步写成 JSONL, 事后复盘/回放用; REC_PATH 由 --record 打开"""
    if not REC_PATH:
        return
    row = dict(t=round(time.time(), 3), event=event)
    try:
        p, rpy = pose(_ARM)
        row["p"] = [round(float(v), 1) for v in p]
        row["rpy"] = [round(float(v), 1) for v in rpy]
        row["q"] = cur_joints()
    except Exception:
        pass
    row.update(kw)
    try:
        with open(REC_PATH, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[record] 写失败: %s" % e)

def wait_still(arm, target=None, tol=2.0, timeout=8.0):
    """轮询法兰位姿, 直到收敛到 target(或不再变化); 超时则刹车"""
    t0, last, still = time.time(), None, 0
    while time.time() - t0 < timeout:
        p, _ = pose(arm)
        err = None if target is None else float(np.linalg.norm(p - np.asarray(target)))
        if err is not None and err < tol:
            return True, err
        if last is not None and float(np.linalg.norm(p - last)) < 0.15:
            still += 1
            if still >= 5:
                if err is not None and err > 2.0:
                    return False, err          # 一动没动却没到目标 = 控制器忽略了这条笛卡尔目标
                return True, err
        else:
            still = 0
        last = p
        time.sleep(0.05)
    soft_stop(arm)
    return False, err

def soft_stop(arm):
    """软停: 把当前关节角当目标连发, 阻尼住不让它继续走"""
    q = _joints(arm)
    if not q or any(v is None for v in q):
        return
    t0 = time.time()
    while time.time() - t0 < 0.3:
        arm.move_j(q)
        time.sleep(0.02)

def _joints(arm):
    j = arm.get_joint_angles()
    return [v for v in (j.msg if j else [0]*6)]

def move_j(arm, q, speed=5, timeout=15.0):
    if not ensure_ctrl(arm):
        return False, None
    arm.set_speed_percent(speed)
    t0 = time.time()
    while time.time() - t0 < 1.0:
        arm.move_j([math.radians(v) for v in q]); time.sleep(0.02)
    # 只转 J6 时法兰位置不变, wait_still 会立刻判"静止", 所以这里直接看关节角
    t0, err = time.time(), None
    while time.time() - t0 < timeout:
        cur = cur_joints()
        if cur:
            err = max(abs(a - b) for a, b in zip(cur, q))
            if err < 1.5:
                record("move_j", target=[round(v, 1) for v in q], ok=True, err=round(err, 2))
                return True, err
        time.sleep(0.05)
    soft_stop(arm)
    record("move_j", target=[round(v, 1) for v in q], ok=False, err=None if err is None else round(err, 2))
    return False, err

def move_p(arm, xyzrpy, speed=5, timeout=10.0, force=False):
    """xyz(mm)+rpy(deg); 连发 0.9s 保证生效, 然后看门狗等收敛

    前置两步检查: ①示教模式 ②可达性(kin.py)。任一不过就不发指令。
    """
    if not ensure_ctrl(arm):
        return False, None
    if not ik_precheck(xyzrpy, force=force):
        record("move_p", target=[round(v, 1) for v in xyzrpy], ok=False, err=None, why="ik")
        return False, None
    arm.set_speed_percent(speed)
    cmd = [xyzrpy[0]/1000, xyzrpy[1]/1000, xyzrpy[2]/1000,
           math.radians(xyzrpy[3]), math.radians(xyzrpy[4]), math.radians(xyzrpy[5])]
    t0 = time.time()
    while time.time() - t0 < 0.9:
        arm.move_p(cmd); time.sleep(0.02)
    ok, err = wait_still(arm, target=xyzrpy[:3], timeout=timeout)
    record("move_p", target=[round(v, 1) for v in xyzrpy], ok=bool(ok),
           err=None if err is None else round(float(err), 2))
    return ok, err

def gripper(eff, width_mm, force_n, seconds=2.0):
    """关键: 夹爪单帧命令无效, 必须连发"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        eff.move_gripper_m(value=width_mm/1000.0, force=force_n); time.sleep(0.02)
    time.sleep(0.4)
    m = eff.get_gripper_status().msg
    w, fo = round(m.value*1000, 1), round(m.force, 2)
    record("grip", cmd_mm=width_mm, cmd_n=force_n, width=w, force=fo)
    return w, fo

# ---------------------------------------------------------------- vision
_CAP = None

def _cap_server():
    """常驻取图进程: 只 init 一次相机, 之后每帧约 0.27s (单次进程要 1.3s)"""
    global _CAP
    if _CAP is not None and _CAP.poll() is None:
        return _CAP
    try:
        p = subprocess.Popen([VISION_PY, CAPTURE, "--rotate", "180", "--serve"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except Exception:
        return None
    t0 = time.time()
    while time.time() - t0 < 20:          # SDK 自己会往 stdout 打日志, 跳过非标记行
        line = p.stdout.readline()
        if not line:
            break
        if line.strip() == "READY":
            _CAP = p
            return p
    try: p.kill()
    except Exception: pass
    return None

def close_capture():
    global _CAP
    if _CAP is not None and _CAP.poll() is None:
        try:
            _CAP.stdin.write("quit\n"); _CAP.stdin.flush()
            _CAP.wait(timeout=3)
        except Exception:
            try: _CAP.kill()
            except Exception: pass
    _CAP = None

atexit.register(close_capture)


def _load_fresh(tag, t0):
    """只接受本次新写出的图, 防止失败时把上一轮的旧文件当成功"""
    f = f"{OUT}/{tag}_color.png"; g = f"{OUT}/{tag}_depth.npy"
    try:
        if not (os.path.exists(f) and os.path.exists(g)): return None, None
        if os.path.getmtime(f) < t0 - 0.05: return None, None
        return cv2.imread(f), np.load(g)
    except Exception:
        return None, None

def capture(tag="auto", retries=2):
    """取一对 RGB-D; 常驻进程坏了/相机偶发无帧时自动退回单次进程并重试"""
    for attempt in range(retries):
        p = _cap_server()
        if p is not None:
            try:
                t0 = time.time()
                p.stdin.write(tag + "\n"); p.stdin.flush()
                while time.time() - t0 < 20:
                    line = p.stdout.readline()
                    if not line:
                        break
                    st = line.strip()
                    if st == "OK":
                        bgr, dep = _load_fresh(tag, t0)
                        if bgr is not None and dep is not None:
                            return bgr, dep
                        break
                    if st.startswith("ERR"):
                        print("[capture] 常驻进程报错:", st); break   # -> 退回单次
            except Exception:
                pass
            close_capture()              # 常驻进程坏了 -> 退回单次
        t1s = time.time()
        r = subprocess.run([VISION_PY, CAPTURE, "--rotate", "180", "--tag", tag],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("[capture] 单次取图失败:", (r.stderr or r.stdout).strip()[-200:])
        bgr, dep = _load_fresh(tag, t1s)
        if bgr is not None and dep is not None:
            return bgr, dep
        if attempt + 1 < retries:
            print("[capture] 重试 %d/%d" % (attempt + 1, retries))
            time.sleep(1.0)
    print("[capture] 取图彻底失败(相机无帧)")
    return None, None

def find_box(bgr, dep):
    """颜色(棕) + 深度双门限 -> 质心像素 + 深度(mm)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 25, 30), (38, 255, 255))
    d = dep.astype(np.float32)
    # 深度只用来剔除远处背景; 白色标签处深度为 0(无回波), 必须保留, 否则质心被拉到盒子下缘
    if np.count_nonzero(d) > 0.02 * d.size:
        m = (d < CFG["depth_max"]) | (d == 0)
        mask = cv2.bitwise_and(mask, m.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 3000: return None
    M = cv2.moments(c)
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    ys, xs = np.where(mask > 0)
    dv = d[ys, xs]
    dv = dv[dv > 0]
    hull = cv2.convexHull(c)
    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(hull)
    P = cv2.boxPoints(((rcx, rcy), (rw, rh), rang))
    e0, e1 = P[1] - P[0], P[2] - P[1]
    le = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    long_vert = bool(abs(le[1]) > abs(le[0]))
    # 短边方向(相对画面水平, 0~180 度): 夹爪沿画面 x 开合, 所以目标是这条边水平
    short_ang = (rang if rw < rh else rang + 90.0) % 180.0
    return {"u": float(cx), "v": float(cy), "short_ang": float(short_ang),
            "depth_mm": float(np.median(dv)) if dv.size else -1.0,
            "rw": float(rw), "rh": float(rh), "rang": float(rang),
            "long_vert": long_vert, "long_len": float(max(rw, rh)),
            "short_len": float(min(rw, rh))}

def cur_joints():
    """取关节角(度)。以 SDK 为主; armraw(CAN 原始帧) 实测偶发读到跳变值, 只作兜底"""
    j = _ARM.get_joint_angles() if _ARM is not None else None
    if j:
        v = [round(math.degrees(x), 2) for x in j.msg]
        if len(v) == 6 and all(abs(x) < 200 for x in v):
            return v
    try:
        v = armraw.read_true(2.0)
        if v and all(abs(x) < 200 for x in v):
            return [round(x, 2) for x in v]
    except Exception:
        pass
    return None

def goto_q(arm, target, steps=3, speed=12):
    """关节空间分段插值, 每步单关节变化 <=35 度"""
    q0 = cur_joints()
    if q0 is None:
        sys.exit("取不到关节角")
    path = [[round(a + (b - a) * k / steps, 2) for a, b in zip(q0, target)] for k in range(1, steps + 1)]
    for q in path:
        mx = max(abs(a - b) for a, b in zip(q0, q))
        print(f"[move_j] -> {q}  (max d={mx:.1f} deg)")
        ok, err = move_j_go(arm, q, speed=speed, timeout=max(15.0, mx / 3.0 * 4.0), label="move_j")
        print(f"[move_j]    ok={ok} err={err}")
        if not ok:
            sys.exit("关节移动未收敛")
        q0 = q

def move_j_go(arm, q, speed=12, timeout=40.0, label=""):
    """一次连贯的关节空间运动: 目标连发 0.8s 锁存, 然后边等边报进度 (不中断/不分段)"""
    if not ensure_ctrl(arm):
        return False, None
    arm.set_speed_percent(speed)
    t0 = time.time()
    while time.time() - t0 < 0.8:
        arm.move_j([math.radians(v) for v in q])
        time.sleep(0.02)
    t0, err, tlast = time.time(), None, 0.0
    while time.time() - t0 < timeout:
        cur = cur_joints()
        if cur:
            err = max(abs(a - b) for a, b in zip(cur, q))
            if err < 1.0:
                print("\r[%s] 到位 err=%.2f° 用时 %.1fs%s" % (label or "move_j", err, time.time()-t0,
                      " " * 20))
                record("move_j_go", target=[round(v, 1) for v in q], ok=True, err=round(err, 2))
                return True, err
            if time.time() - tlast > 0.4:
                tlast = time.time()
                print("\r[%s] 剩余 %.1f°  已用 %.1fs" % (label or "move_j", err, time.time()-t0), end="")
        time.sleep(0.05)
    print()
    soft_stop(arm)
    record("move_j_go", target=[round(v, 1) for v in q], ok=False,
           err=None if err is None else round(err, 2))
    return False, err


def goto_home(arm, speed=20, precheck=True, force=False, safe_z=240.0):
    """连贯回零: (必要时先抬高) -> 仿真预检整条关节路径 -> 一段到底"""
    q0 = cur_joints()
    if q0 is None:
        print("[home] 取不到关节角")
        return False
    p, rpy = pose(arm)
    if p is not None and p[2] < safe_z:
        dz = safe_z - p[2]
        print("[home] 工具太低(%.0f mm)，先沿 +Z 抬 %.0f mm" % (p[2], dz))
        ok, _ = move_p(arm, [p[0], p[1], p[2] + dz] + list(rpy), speed=10, timeout=25)
        if not ok:
            print("[home] 抬升失败，停止")
            return False
        q0 = cur_joints()
    if precheck:
        try:
            import simpath
            r = simpath.check_path(q0, CFG["home_q"], n=24, verbose=False)
            print("[home] 仿真预检：桌面最小间隙 %.1f mm，末端最低 z %.1f mm，越限 %s" % (
                r["min_table"], r["min_flange_z"], r["bad_limits"] or "无"))
            if not r["ok"] and not force:
                print("[home] 这条直线路径不安全，已放弃（可加 --force 强行走）")
                return False
        except Exception as ex:
            print("[home] 仿真预检跳过: %s" % ex)
    mx = max(abs(a - b) for a, b in zip(q0, CFG["home_q"]))
    print("[home] 一段走到零位，最大关节变化 %.1f°，速度 %d%%" % (mx, speed))
    ok, err = move_j_go(arm, CFG["home_q"], speed=speed, timeout=max(30.0, mx / 3.0 * 4.0), label="home")
    if ok:
        p, rpy = pose(arm)
        print("[home] 完成 法兰 %s mm rpy %s" % (np.round(p, 1), np.round(rpy, 1)))
    return ok


def goto_xy(arm, xy, max_step=30.0, speed=12):
    """在固定高度把法兰平移到 xy(mm), 分步避免大跳"""
    base, rpy = pose(arm)
    d = np.array(xy, dtype=float) - base[:2]
    n = max(1, int(np.ceil(np.linalg.norm(d) / max_step)))
    for i in range(1, n + 1):
        p = base[:2] + d * i / n
        ok, _ = move_p(arm, [p[0], p[1], base[2], *rpy], speed=speed, timeout=8)
        if not ok:
            print(f"[search] 到 ({p[0]:.0f},{p[1]:.0f}) 没走通")
            return False
    return True

def search_box(arm):
    """盒子不在视野里: 以观察位姿为中心螺旋扫一圈 (±170mm 覆盖整张垫子)"""
    base, rpy = pose(arm)
    anchor = base[:2].copy()
    bgr, dep = capture("search0")
    b = find_box(bgr, dep)
    if b:
        print("[search] 首帧就能看到盒子"); return b
    for r, n in ((90, 8), (170, 8), (240, 6)):
        for k in range(n):
            ang = 2 * math.pi * k / n - math.pi / 2
            dx, dy = r * math.cos(ang), r * math.sin(ang)
            if not goto_xy(arm, anchor + np.array([dx, dy])):
                continue
            bgr, dep = capture(f"search_{int(r)}_{k}")
            b = find_box(bgr, dep)
            if b:
                print(f"[search] 在 ({dx:+.0f},{dy:+.0f}) mm 处找到盒子"); return b
        print(f"[search] 半径 {r:.0f} 一圈没有")
        goto_xy(arm, anchor)
    return None

def goto_obs(arm):
    """回示教观察位姿(关节空间, 不用长距离笛卡尔)"""
    goto_q(arm, CFG["obs_q"])

# ---------------------------------------------------------------- calib
def calib(arm):
    """测像素误差 -> 基座位移 的 2x2 雅可比 (mm/px)"""
    print("[calib] 走观察位姿")
    goto_obs(arm)
    time.sleep(0.4)
    bgr, dep = capture("calib0")
    b0 = find_box(bgr, dep)
    if not b0: sys.exit("calib: 看不到盒子")
    base, rpy = pose(arm)
    J = np.zeros((2, 2))
    for i, axis in enumerate((0, 1)):
        tgt = base.copy(); tgt[axis] += 20.0
        ok, _ = move_p(arm, [*tgt, *rpy], speed=5)
        if not ok: sys.exit("calib: 移动失败")
        time.sleep(0.4)
        bgr, dep = capture(f"calib{i+1}")
        b = find_box(bgr, dep)
        if not b: sys.exit("calib: 移动后看不到盒子")
        # 像素位移 对 基座 +20mm 的响应 => 每 mm 的像素变化
        J[:, i] = [(b["u"]-b0["u"])/20.0, (b["v"]-b0["v"])/20.0]
        move_p(arm, [*base, *rpy], speed=5)
        time.sleep(0.4)
    Jinv = np.linalg.inv(J)
    # 角度增益: J6 转 +10 度, 看矩形角度变化多少
    bgr, dep = capture("calib_yaw0"); b0 = find_box(bgr, dep)
    if not b0: sys.exit("calib: 测角度增益时看不到盒子")
    q = cur_joints(); q[5] = round(q[5] + 10.0, 2)
    move_j(arm, q, speed=5, timeout=20)
    time.sleep(0.4)
    bgr, dep = capture("calib_yaw1"); b1 = find_box(bgr, dep)
    if not b1: sys.exit("calib: 转动后看不到盒子")
    dyaw = (rect_err(b1["rang"]) - rect_err(b0["rang"]) + 45.0) % 90.0 - 45.0
    tgain = dyaw / 10.0
    print(f"[calib] J6 +10deg -> 矩形角度变化 {dyaw:+.2f} deg => yaw_gain={tgain:+.3f} deg/deg")
    if abs(tgain) < 0.2: sys.exit("calib: 角度增益太小, 检查 J6 是否真的在转")
    q = cur_joints(); print("[calib] J6 before restore:", q[5])
    q[5] = round(q[5] - 10.0, 2)
    move_j(arm, q, speed=5, timeout=20)
    time.sleep(0.5)
    print("[calib] J6 after restore:", cur_joints()[5])
    json.dump({"J": J.tolist(), "Jinv": Jinv.tolist(), "target_px": CFG["target_px"],
               "yaw_gain": tgain, "calib_j6": round(cur_joints()[5], 2)},
              open(CALIB, "w"), indent=2)
    print("[calib] J(px/mm)=", J.round(3).tolist(), " -> ", CALIB)

def rect_err(rang):
    """minAreaRect 的角度(度) -> 相对画面轴对齐的偏差, 落在 (-45, 45]"""
    a = rang % 90.0
    if a > 45.0: a -= 90.0
    return a

def short_err(ang):
    """短边相对水平线的偏差, 落在 (-90, 90]: 唯一目标(不需要在 0/90 之间试错)"""
    a = ang % 180.0
    if a > 90.0: a -= 180.0
    return a

def align_yaw(arm, tol=None, max_iter=10):
    """只转 J6, 把盒子短边转到画面水平(夹爪沿画面 x 开合)。
    增益现场实测(模型里 rang 的约定不可靠, 标定值在不同位姿会反号), 再按实测增益闭环。"""
    tol = CFG["yaw_tol"] if tol is None else tol
    q0 = cur_joints()
    if q0 is None: sys.exit("取不到关节角")

    def meas(tag):
        bgr, dep = capture(tag)
        b = find_box(bgr, dep)
        if not b: return None, None
        return b, short_err(b["short_ang"])

    b, e0 = meas("yaw_p0")
    if b is None: sys.exit("角度伺服: 看不到盒子")
    print(f"[yaw] 起始 矩形={b['rw']:.0f}x{b['rh']:.0f} 短边偏差={e0:+.1f} deg")

    # 试探 12 度, 现场测增益
    probe = 12.0
    q = list(q0); q[5] = round(q0[5] + probe, 2)
    if not move_j(arm, q, speed=8, timeout=25)[0]: sys.exit("角度试探未收敛")
    b, e1 = meas("yaw_p1")
    if b is None: sys.exit("角度伺服: 试探后看不到盒子")
    g = short_err(e1 - e0) / probe
    print(f"[yaw] J6 {probe:+.0f} deg -> 偏差 {e0:+.1f} -> {e1:+.1f}, 实测增益 {g:+.2f} deg/deg")
    if abs(g) < 0.3:
        print("[yaw] 增益异常, 放弃"); return False

    delta = probe
    for it in range(max_iter):
        b, a = meas("yaw")
        if b is None:
            delta = round(delta - probe * 0.5, 2)
            q = list(q0); q[5] = round(q0[5] + delta, 2)
            print(f"[yaw {it}] 看不到盒子, 回退到 J6 {q[5]:.1f}")
            move_j(arm, q, speed=8, timeout=25); continue
        print(f"[yaw {it}] 矩形={b['rw']:.0f}x{b['rh']:.0f} 长边{'竖' if b['long_vert'] else '横'}"
              f" 短边偏差={a:+.1f} deg (gain {g:+.2f})")
        if abs(a) < tol:
            print(f"[yaw] 收敛 矩形={b['rw']:.0f}x{b['rh']:.0f} 长边{'竖 ✔' if b['long_vert'] else '横 ✘'}")
            ar = b["long_len"] / max(b["short_len"], 1e-6)
            if not b["long_vert"] and ar > 1.12:
                print(f"[yaw] 警告: 长边是横的(长宽比 {ar:.2f}), 闭合可能夹在长边上")
            return True
        d = -a / g * 0.9
        d = max(-20.0, min(20.0, d))
        if abs(q0[5] + delta + d) > 115:
            print(f"[yaw] J6 将到 {q0[5]+delta+d:.0f} 度, 越限, 停止"); return False
        delta = round(delta + d, 2)
        q = list(q0); q[5] = round(q0[5] + delta, 2)
        print(f"[yaw]    J6 {q0[5]:.1f} {delta:+.1f} -> {q[5]:.1f}")
        if not move_j(arm, q, speed=8, timeout=25)[0]: sys.exit("角度伺服: 转动未收敛")
    return False

def align(arm, tol=10.0, max_iter=20):
    """视觉伺服: 把盒子质心移到 target_px。
    注意: 像素轴随 J6 转, 所以先用 yaw_gain 把像素误差转回标定时的相机朝向;
          像素增益还随距离变, 所以带线搜索(误差变大就减半步长)"""
    cfg = json.load(open(CALIB))
    J = np.array(cfg["Jinv"])
    tgt = np.array(CFG["target_px"])
    j6_now = cur_joints()[5] if cur_joints() else cfg.get("calib_j6", 0.0)
    theta = math.radians(cfg.get("yaw_gain", 1.0) * (j6_now - cfg.get("calib_j6", 0.0)))
    # 相机随 J6 转了 ΔJ6, 像素轴随之旋转; 同一个基座位移在旋转后的像素系里要反向旋转回来
    Rm = np.array([[ math.cos(theta), math.sin(theta)],
                   [-math.sin(theta), math.cos(theta)]])
    print(f"[servo] J6={j6_now:.1f} 相对标定转了 {j6_now - cfg.get('calib_j6', 0.0):+.1f} deg"
          f" -> 像素轴旋转 {math.degrees(theta):+.1f} deg")
    prev, step_mm = None, CFG["max_lateral_mm"]
    for it in range(max_iter):
        bgr, dep = capture("servo")
        b = find_box(bgr, dep)
        if not b:
            print("[servo] 看不到盒子"); return False
        err = tgt - np.array([b["u"], b["v"]])
        n = float(np.linalg.norm(err))
        print(f"[servo {it}] 质心=({b['u']:.0f},{b['v']:.0f}) 误差={err.round(1)} px |e|={n:.1f} 步长上限={step_mm:.0f}mm")
        if n < tol:
            print("[servo] 对中完成"); return True
        if prev is not None and abs(n - prev) < 1.0:
            print("[servo] 质心纹丝不动: 该位置够不着(笛卡尔目标被控制器忽略)"); return False
        if prev is not None:
            if n > prev * 1.02:
                step_mm = max(5.0, step_mm * 0.5)
                print(f"[servo]   误差变大, 步长降到 {step_mm:.0f}mm")
            elif n < prev * 0.7 and step_mm < CFG["max_lateral_mm"]:
                step_mm = min(CFG["max_lateral_mm"], step_mm * 1.5)
                print(f"[servo]   收敛良好, 步长回到 {step_mm:.0f}mm")
        prev = n
        step = J @ (Rm @ err)
        sn = float(np.linalg.norm(step))
        if sn > step_mm: step *= step_mm / sn
        base, rpy = pose(arm)
        ok, me = move_p(arm, [base[0]+step[0], base[1]+step[1], base[2], *rpy], speed=8)
        if not ok:
            mtxt = f"{me:.1f}mm" if isinstance(me, (int, float)) else "未知"
            print(f"[servo] 走不动 (残差 {mtxt}, 到关节限位/臂展边缘)"); return False
        time.sleep(0.05)
    return False

def align_adaptive(arm, tol=10.0):
    """够不着就压低高度再试: 同样的 xy, 高度越低需要的 J5 越小"""
    if align(arm, tol=tol): return True
    # 指尖在法兰下方 61~138mm; 法兰降到 148mm 时指尖仍比盒顶高约 40mm, 不会磕
    for dz in (28, 24, 20, 17, 14):
        base, rpy = pose(arm)
        nz = max(CFG["grasp_z"], base[2] - dz)
        if nz > base[2] - 1.0:
            print(f"[retry] 已经压到 {base[2]:.0f}mm(=抓取高度), 再低就碰盒子, 放弃")
            return False
        print(f"[retry] 该高度够不着, 法兰降到 {nz:.0f}mm 再试"
              f" (此刻 xy=({base[0]:.0f},{base[1]:.0f}) 半径 {np.hypot(base[0],base[1]):.0f}mm)")
        ok, _ = move_p(arm, [base[0], base[1], nz, *rpy], speed=6, timeout=12)
        if ok and align(arm, tol=tol): return True
    return False

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--home", action="store_true", help="回零位(关节空间 3 段)")
    ap.add_argument("--no-search", action="store_true", help="跳过自动搜索")
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="调试: 对中后先把 J6 转这么多度, 模拟盒子被转过")
    ap.add_argument("--record", nargs="?", const="auto", default=None,
                    help="把每步写 JSONL(out/run_<时间>.jsonl), 便于复盘")
    a = ap.parse_args()

    global REC_PATH
    if a.record:
        REC_PATH = (os.path.join(OUT, "run_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
                    if a.record == "auto" else a.record)
        print("[record] %s" % REC_PATH)

    arm, eff = connect()
    base, rpy = pose(arm)
    print(f"[init] 法兰 {base.round(1)} rpy {rpy.round(1)}")
    gripper(eff, CFG["open_mm"], 1.0)

    if a.calib:
        calib(arm); return

    if a.release:
        # 降回抓取高度 -> 张开 -> 抬走
        cur = base.copy()
        while cur[2] > CFG["grasp_z"] + 0.5:
            cur[2] = max(CFG["grasp_z"], cur[2] - CFG["step_mm"])
            move_p(arm, [*cur, *rpy], speed=CFG["speed"])
        gripper(eff, CFG["open_mm"], 1.0)
        up = cur.copy(); up[2] += CFG["lift_mm"]
        move_p(arm, [*up, *rpy], speed=CFG["speed"])
        print("[release] 已放下"); return

    # 0) 归位
    if a.home:
        goto_q(arm, CFG["home_q"], steps=3, speed=5)
        p, r = pose(arm)
        print(f"[home] 法兰 ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
        return

    # 1) 观察位姿(关节空间, 从 home 长距离笛卡尔会进奇异区)
    goto_obs(arm)
    # 1.5) 盒子可能不在视野里: 螺旋扫一圈
    if not search_box(arm): sys.exit("视野里找不到盒子(扫过 ±170mm)")
    # 2) 先把盒子拉到画面中央(避免转腕时出画), 再交替: 角度 -> 位置 -> 角度 -> 位置
    if not align_adaptive(arm): sys.exit("对中失败")
    if abs(a.yaw_offset) > 1e-6:
        q = cur_joints(); q[5] = round(q[5] + a.yaw_offset, 2)
        if abs(q[5]) > 115: sys.exit(f"--yaw-offset 会让 J6 到 {q[5]:.0f}, 越限")
        print(f"[dbg] 模拟盒子被转: J6 {cur_joints()[5]:.1f} -> {q[5]:.1f}")
        move_j(arm, q, speed=5, timeout=20); time.sleep(0.4)
    ok_yaw = ok_xy = False
    for rnd in range(2):
        ok_yaw = align_yaw(arm)
        ok_xy = align_adaptive(arm, tol=8.0)
        if ok_yaw and ok_xy: break
    if not ok_yaw: sys.exit("角度对齐失败")
    if not ok_xy: sys.exit("对中失败")
    # 3) 分段下降 (每次 10mm, 每段校验 FK)
    base, rpy = pose(arm)
    cur = base.copy()
    while cur[2] > CFG["grasp_z"] + 0.5:
        far = cur[2] - CFG["grasp_z"] > 45.0     # 离盒子还远就大步快走, 最后 45mm 慢下来
        cur[2] = max(CFG["grasp_z"], cur[2] - (30.0 if far else CFG["step_mm"]))
        ok, err = move_p(arm, [*cur, *rpy], speed=10 if far else CFG["speed"])
        p, _ = pose(arm)
        print(f"[down] 目标 z={cur[2]:.0f} 到位 z={p[2]:.1f} err={err}")
        if not ok or abs(p[2]-cur[2]) > 2.5:
            print("[down] 未收敛, 抬回并退出"); cur[2] += 40; move_p(arm, [*cur, *rpy]); sys.exit(1)
    # 4) 闭合 + 校验
    w0, f0 = gripper(eff, CFG["close_mm"], CFG["close_n"])
    print(f"[grip] 闭合后 宽={w0}mm 力={f0}N")
    if w0 < CFG["close_mm"] + 2 or abs(f0) < 0.5:
        print("[grip] 没夹到东西, 张开退出"); gripper(eff, CFG["open_mm"], 1.0); sys.exit(2)
    # 5) 抬起 + 校验
    #    盒子在臂展边缘时"正上方"可能 IK 无解(抬高等于加大水平伸出量), 退化成边回收边抬
    p0, rpy = pose(arm)
    up = p0.copy(); up[2] += CFG["lift_mm"]
    ok, _ = move_p(arm, [*up, *rpy], speed=CFG["speed"])
    if not ok:
        print("[lift] 正上方不可达(臂展边缘), 改走\"边回收边抬\"")
        for frac, dz in ((0.06, 16.0), (0.12, 34.0), (0.18, 50.0)):
            t = [p0[0] * (1 - frac), p0[1] * (1 - frac), p0[2] + dz, *rpy]
            ok, _ = move_p(arm, t, speed=CFG["speed"], timeout=20)
            print(f"[lift] 回收 {frac*100:.0f}% 抬到 z={t[2]:.0f} ok={ok}")
            if not ok: break
    w1, f1 = gripper(eff, CFG["close_mm"], CFG["close_n"], seconds=0.4)
    print(f"[lift] 抬起后 宽={w1}mm 力={f1}N")
    print("[done] " + ("抓取成功" if abs(f1) > 0.5 else "抓取后掉落"))

if __name__ == "__main__":
    main()