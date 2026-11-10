#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录制机械臂关节流（给 ACT 采数据用）。

用 /home/po/arm-env/bin/python 跑（有 pyAgxArm）。

    python rec_arm.py --out act_data/ep_0000 --seconds 40 --hz 100

产出：<out>/arm.csv（t, q1..q6, grip_mm, ok）
只读，不发任何运动指令，所以示教模式下也可以安全录。Ctrl-C 会正常收尾。
"""
import argparse, csv, os, sys, time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autograsp as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--hz", type=float, default=100.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arm, eff = A.connect()
    time.sleep(0.5)
    rows = []
    t_start = time.time()
    dt = 1.0 / max(10.0, args.hz)
    grip = 0.0
    last_grip = 0.0
    n_bad = 0
    try:
        while time.time() - t_start < args.seconds:
            q = arm.get_joint_angles()
            now = time.time() - t_start
            if q is None:
                n_bad += 1
                time.sleep(dt)
                continue
            v = [math.degrees(x) for x in q.msg]
            if len(v) != 6:
                n_bad += 1
                continue
            # 夹爪开度读得慢，20Hz 足够，其余时间保持
            if now - last_grip > 0.05:
                try:
                    grip = round(float(eff.get_gripper_status().msg.value) * 1000.0, 1)
                except Exception:
                    pass
                last_grip = now
            rows.append([round(now, 4)] + [round(x, 4) for x in v] + [grip, 1])
            time.sleep(dt)
    except KeyboardInterrupt:
        print("rec_arm: 收到中断，收尾", flush=True)

    with open(os.path.join(args.out, "arm.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "j1", "j2", "j3", "j4", "j5", "j6", "grip", "ok"])
        w.writerows(rows)
    print("arm done: %d samples / %.1fs (bad %d)" % (len(rows), time.time() - t_start, n_bad), flush=True)


if __name__ == "__main__":
    main()
