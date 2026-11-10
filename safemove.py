#!/usr/bin/env python3
"""安全关节移动: 低速 + 误差收敛 + 看门狗中止。

用法: safemove.py <delta1> ... <delta6> [--speed 5] [--tol 0.5] [--timeout 8]
      在上一次成功到位后再次调用可依次串行。
"""
import time, math, sys, json
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

D = math.pi / 180.0
CHANNEL = "can0"


def connect():
    cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel=CHANNEL)
    arm = AgxArmFactory.create_arm(cfg)
    arm.connect()
    time.sleep(1.2)
    arm.enable()
    time.sleep(0.5)
    return arm


def soft_stop(arm, speed=20):
    """平滑停住: 把目标改成当前位姿（SDK 没有 stop()）"""
    cur = read_joints(arm)
    if cur:
        arm.set_speed_percent(speed)
        for _ in range(10):
            arm.move_j([x * D for x in cur])
            time.sleep(0.02)


def read_joints(arm):
    # SDK 的 get_joint_angles() 在某些状态下读到过期值，改用 CAN 原始帧
    try:
        from armraw import read_true
        q = read_true(0.6)
        if q is not None:
            return q
    except Exception:
        pass
    ja = arm.get_joint_angles()
    if ja is None:
        return None
    return [x / D for x in ja.msg]


def move_to(arm, target, speed=5, tol=0.5, timeout=8.0):
    """流式发目标，误差收敛+看门狗。返回 (ok, 最终关节, 最大误差轨迹)"""
    arm.set_speed_percent(speed)
    t0 = time.time()
    best = 1e9
    last_improve = time.time()
    trace = []
    while time.time() - t0 < timeout:
        cur = read_joints(arm)
        if cur is None:
            time.sleep(0.05)
            continue
        err = max(abs(a - b) for a, b in zip(cur, target))
        trace.append(round(err, 2))
        if err < tol:
            arm.move_j([x * D for x in target])
            time.sleep(0.15)
            cur = read_joints(arm)
            err = max(abs(a - b) for a, b in zip(cur, target)) if cur else 999
            return err < max(tol * 3, 1.5), cur, trace
        if err < best - 0.2:
            best = err
            last_improve = time.time()
        elif time.time() - last_improve > 1.5:
            soft_stop(arm)
            return False, cur, trace + ["STUCK"]
        arm.move_j([x * D for x in target])
        time.sleep(0.02)
    soft_stop(arm)
    return False, read_joints(arm), trace + ["TIMEOUT"]


def main():
    args = [a for a in sys.argv[1:]]
    speed = 5
    if "--speed" in args:
        i = args.index("--speed"); speed = int(args[i + 1]); del args[i:i + 2]
    timeout = 8.0
    if "--timeout" in args:
        i = args.index("--timeout"); timeout = float(args[i + 1]); del args[i:i + 2]
    vals = [float(v) for v in args[:6]]
    arm = connect()
    cur = read_joints(arm)
    if cur is None:
        print(json.dumps({"ok": False, "why": "no joint feedback"})); return
    target = vals if "--abs" in args else [c + d for c, d in zip(cur, vals)]
    if "--abs" in args:
        big = max(abs(a - c) for a, c in zip(target, cur))
        if big > 40 and speed > 8:
            print(json.dumps({"ok": False, "why": "big move %0.1fdeg needs speed<=8" % big})); return
    ok, final, trace = move_to(arm, target, speed=speed, timeout=timeout)
    print(json.dumps({
        "ok": bool(ok),
        "from": [round(v, 2) for v in cur],
        "target": [round(v, 2) for v in target],
        "final": None if final is None else [round(v, 2) for v in final],
        "max_err": max([t for t in trace if isinstance(t, float)]) if any(isinstance(t, float) for t in trace) else None,
        "trace_tail": trace[-6:],
    }))


if __name__ == "__main__":
    main()