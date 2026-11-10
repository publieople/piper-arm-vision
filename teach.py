#!/usr/bin/env python3
"""teach.py —— 拖动示教：录制 / 回放

示教 = 机械臂进入重力补偿的"拖动模式"，人手拖着走；我们把关节角一路记下来，
再按时间轴用 move_j 重新发一遍。

  teach.py info                 看当前模式（TEACHING / CAN_CTRL）、示教状态
  teach.py rec [--hz 100]       开始录制（录制期间你拖动机械臂），Ctrl-C 结束
  teach.py play [轨迹文件] [--speed 1.0] [--loop 1]
  teach.py list                 列出已录轨迹
  teach.py fw start|stop|play|pause|cont|abort|goto   固件自带的示教记录/复现

说明
----
* 录制的是 6 个关节角（度）+ 时间戳，100Hz 左右；夹爪开度不动，回放时不重放夹爪。
* 回放走关节空间（move_j 流式），不会经过笛卡尔 IK，所以不会有奇异点甩臂问题；
  但仍会检查关节限位和"实际角度偏离目标 >30° 就停"。
* 回放前会自动把控制器从 TEACHING_MODE 切回 CAN_CTRL（需要人手按一下示教按钮，
  按了我们才发得动指令）。
* 固件自带示教（fw 子命令）把轨迹存在控制器里，可以脱离电脑复现；坑是它把
  "示教按钮"和"录制开关"两件事混在一起，且轨迹长度受控制器内存限制。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autograsp as A
import kin

DIR = "/home/po/vision/calib"
LIM_LO = kin.LO
LIM_HI = kin.HI
NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def joint_list(arm):
    j = arm.get_joint_angles()
    if j is None:
        return None
    v = [round(math.degrees(x), 3) for x in j.msg]
    return v if len(v) == 6 else None


def mode_str(arm):
    try:
        m = arm.get_arm_status().msg
        return str(m.ctrl_mode), str(m.teach_status), int(getattr(m, "trajectory_num", -1))
    except Exception as e:
        return ("?", "?", -1)


# ------------------------------------------------------------------ info
def cmd_info(arm, eff):
    ctrl, teach, traj = mode_str(arm)
    q = joint_list(arm)
    print("控制模式   : %s" % ctrl)
    print("示教状态   : %s" % teach)
    print("轨迹点计数 : %d" % traj)
    print("关节(度)   : %s" % (" ".join("%s=%6.2f" % (n, v) for n, v in zip(NAMES, q)) if q else "读不到"))
    print("提示：TEACHING_MODE 下可以直接用手拖；要发指令必须先按一下机械臂上的示教按钮退出。")


# ------------------------------------------------------------------ rec
def cmd_rec(arm, eff, hz=100.0, out=None, seconds=600.0):
    ctrl, teach, _ = mode_str(arm)
    if "TEACHING" not in ctrl:
        print("!! 当前不是示教模式（%s）。先按住机械臂上的示教按钮拖到示教模式，再录。" % ctrl)
        print("   仍然继续录制 10 秒，如果你马上就拖，也能抓到。")
    out = out or os.path.join(DIR, "teach_%s.json" % now_stamp())
    dt = 1.0 / max(10.0, hz)
    traj, ts = [], []
    t0 = time.time()
    warned = False
    print("[rec] 开始录制，拖着机械臂走；录完按 Ctrl-C（或直接按示教按钮退出示教模式）")
    try:
        while True:
            t = time.time() - t0
            q = joint_list(arm)
            if q is None:
                time.sleep(dt)
                continue
            moved = (not traj) or max(abs(a - b) for a, b in zip(q, traj[-1])) >= 0.02
            if moved or (t - ts[-1]) >= 0.1:
                traj.append(q)
                ts.append(round(t, 3))
            if not warned and t > 1.0:
                c2, _, _ = mode_str(arm)
                if "TEACHING" not in c2:
                    print("[rec] 已退出示教模式（%s），自动结束录制" % c2)
                    break
                warned = True
            if t > seconds:
                print("[rec] 到 %.0f 秒了，自动结束" % seconds)
                break
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[rec] 手动结束")
    dur = ts[-1] - ts[0] if ts else 0.0
    if len(traj) < 5 or dur < 0.5:
        print("[rec] 数据太少（%d 点 / %.1f 秒），不保存" % (len(traj), dur))
        return 1
    data = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": len(traj),
            "duration": round(dur, 3), "q": traj, "t": ts}
    os.makedirs(DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    step = max(max(abs(a - b) for a, b in zip(traj[i], traj[i - 1])) for i in range(1, len(traj)))
    rng = [max(c[k] for c in traj) - min(c[k] for c in traj) for k in range(6)]
    print("[rec] 保存 %s" % out)
    print("      %d 点 / %.2f 秒 / 最大单步 %.2f 度" % (len(traj), dur, step))
    print("      各关节活动范围: %s" % " ".join("%s %5.1f" % (n, r) for n, r in zip(NAMES, rng)))
    if step > 6.0:
        print("      !! 有单步超过 6 度，回放时可能有点冲，建议 --speed 0.5")
    return 0


# ------------------------------------------------------------------ play
def cmd_play(arm, eff, path, speed=1.0, loops=1, start=0.0, end=1.0):
    if path is None:
        cands = sorted(p for p in os.listdir(DIR) if p.startswith("teach_") and p.endswith(".json"))
        if not cands:
            print("没有录过轨迹")
            return 1
        path = os.path.join(DIR, cands[-1])
        print("[play] 用最新的轨迹 %s" % path)
    d = json.load(open(path))
    q = np.array(d["q"], float)
    t = np.array(d["t"], float) - d["t"][0]
    dur = float(t[-1])
    t = t * (end - start) + start * dur
    keep = (t >= start * dur) & (t <= end * dur)
    t, q = t[keep], q[keep]
    if len(t) < 3:
        print("轨迹太短")
        return 1
    print("[play] %s  %.2f 秒  %d 点  速度 x%.2f" % (os.path.basename(path), t[-1] - t[0], len(t), speed))
    bad = [NAMES[k] for k in range(6) if q[:, k].min() < LIM_LO[k] or q[:, k].max() > LIM_HI[k]]
    if bad:
        print("!! 轨迹超出关节限位: %s" % ",".join(bad))
        return 2
    if not A.ensure_ctrl(arm):
        print("!! 还在示教模式，先按一下示教按钮再回放")
        return 3
    arm.set_speed_percent(30)
    q0 = np.array(joint_list(arm), float)
    print("[play] 起点偏差 %s 度" % np.round(q[0] - q0, 1).tolist())
    print("[play] 3 秒后开始，手离开工作区")
    time.sleep(3.0)
    tgt_t = t[0]
    i = 0
    wall0 = time.time()
    try:
        for _ in range(loops):
            i = 0
            while i < len(t):
                elapsed = (time.time() - wall0) * speed
                target_t = t[0] + elapsed
                # 找当前应处的插值点
                j = int(np.searchsorted(t, target_t))
                j = min(max(j, 1), len(t) - 1)
                a = (target_t - t[j - 1]) / max(1e-6, (t[j] - t[j - 1]))
                a = float(min(max(a, 0.0), 1.0))
                qs = q[j - 1] + (q[j] - q[j - 1]) * a
                arm.move_j([math.radians(v) for v in qs])
                if i % 50 == 0:
                    real = joint_list(arm)
                    if real is not None:
                        dev = max(abs(a2 - b2) for a2, b2 in zip(real, qs))
                        if dev > 30.0:
                            print("\n[play] 实际角度偏离目标 %.1f 度，停止" % dev)
                            A.soft_stop(arm)
                            return 4
                i += 1
                time.sleep(0.01)
            wall0 = time.time()
        # 稳住最后一个点
        for _ in range(60):
            arm.move_j([math.radians(v) for v in q[-1]])
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[play] 手动中断")
        A.soft_stop(arm)
        return 5
    print("[play] 完成，当前关节 %s" % joint_list(arm))
    return 0


# ------------------------------------------------------------------ fw
FW = {"start": 0x01, "stop": 0x02, "play": 0x03, "pause": 0x04,
      "cont": 0x05, "abort": 0x06, "goto": 0x07}


def cmd_fw(arm, eff, what):
    from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_motion_ctrl import ArmMsgMotionCtrl
    v = FW[what]
    for _ in range(5):
        arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=v))
        time.sleep(0.06)
    time.sleep(0.4)
    ctrl, teach, traj = mode_str(arm)
    print("[fw] 已发 grag_teach_ctrl=0x%02X (%s) → 模式 %s 示教状态 %s 轨迹点 %d"
          % (v, what, ctrl, teach, traj))


def cmd_list():
    if not os.path.isdir(DIR):
        print("没有轨迹")
        return 0
    rows = [p for p in sorted(os.listdir(DIR)) if p.startswith("teach_") and p.endswith(".json")]
    for p in rows:
        try:
            d = json.load(open(os.path.join(DIR, p)))
            print("%-28s %6.2f 秒 %4d 点  %s" % (p, d.get("duration", 0), d.get("samples", 0), d.get("created", "")))
        except Exception:
            print("%-28s (坏文件)" % p)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    r = sub.add_parser("rec")
    r.add_argument("--hz", type=float, default=100.0)
    r.add_argument("--out")
    r.add_argument("--seconds", type=float, default=600.0)
    p = sub.add_parser("play")
    p.add_argument("path", nargs="?")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", type=int, default=1)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=1.0)
    sub.add_parser("list")
    f = sub.add_parser("fw")
    f.add_argument("what", choices=sorted(FW))
    args = ap.parse_args()

    if args.cmd == "list":
        return cmd_list()

    arm, eff = A.connect()
    try:
        if args.cmd == "info":
            return cmd_info(arm, eff)
        if args.cmd == "rec":
            return cmd_rec(arm, eff, hz=args.hz, out=args.out, seconds=args.seconds)
        if args.cmd == "play":
            return cmd_play(arm, eff, args.path, speed=args.speed, loops=args.loop,
                            start=args.start, end=args.end)
        if args.cmd == "fw":
            return cmd_fw(arm, eff, args.what)
    finally:
        A.close_capture()


if __name__ == "__main__":
    sys.exit(main())
