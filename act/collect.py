#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集一条 ACT 演示轨迹。

同时启动两个子进程（相机用 vision-env、机械臂用 arm-env，因为两个 SDK 的
Python 版本不同），各自带时间轴落盘，录完按时间戳合并成 ep.npz。

两种用法：
  1) 人手示教（按机械臂上的示教按钮拖动）：
       python collect.py --seconds 40
  2) 录脚本跑出来的动作（自动生成演示数据）：
       python collect.py --run "/home/po/arm-env/bin/python /home/po/vision/bodygrasp.py"

产出：act_data/ep_XXXX/{cam.mp4, cam_ts.csv, arm.csv, ep.npz}
"""
import argparse, csv, os, signal, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VISION_PY = "/home/po/vision-env/bin/python"
ARM_PY = "/home/po/arm-env/bin/python"


def next_ep(root, name=None):
    os.makedirs(root, exist_ok=True)
    if name:
        return os.path.join(root, name)
    i = 1
    while True:
        d = os.path.join(root, "ep_%04d" % i)
        if not os.path.exists(d):
            return d
        i += 1


def load_cam(path):
    t = []
    with open(path) as fp:
        r = csv.reader(fp)
        next(r)
        for row in r:
            if len(row) < 2:
                continue
            t.append(float(row[0]))
    return np.array(t)


def load_arm(path):
    rows = []
    with open(path) as fp:
        r = csv.reader(fp)
        next(r)
        for row in r:
            if len(row) < 9:
                continue
            rows.append([float(x) for x in row[:9]])
    a = np.array(rows)
    return a[:, 0], a[:, 1:7], a[:, 7]


def merge(epdir, fps):
    ct = load_cam(os.path.join(epdir, "cam_ts.csv"))
    at, aq, ag = load_arm(os.path.join(epdir, "arm.csv"))
    if len(ct) < 5 or len(at) < 5:
        raise RuntimeError("样本太少：cam %d 帧 / arm %d 点" % (len(ct), len(at)))
    q = np.stack([np.interp(ct, at, aq[:, k]) for k in range(6)], axis=1)
    g = np.interp(ct, at, ag)
    np.savez_compressed(os.path.join(epdir, "ep.npz"),
                        t=ct.astype(np.float32), q=q.astype(np.float32),
                        grip=g.astype(np.float32), fps=float(fps),
                        n_frames=np.int32(len(ct)))
    print("合并：%d 帧 / %.1fs  qmin=%s qmax=%s"
          % (len(ct), ct[-1] - ct[0], np.round(q.min(0), 1).tolist(),
             np.round(q.max(0), 1).tolist()), flush=True)


def stop_all(procs):
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)   # 录制进程走 KeyboardInterrupt 分支收尾
    for p in procs:
        try:
            p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/po/vision/act_data")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--name")
    ap.add_argument("--no-wait", action="store_true", help="立刻开始，不等回车")
    ap.add_argument("--run", help="录这段时间里跑这条命令（跑完就停录）")
    args = ap.parse_args()

    ep = next_ep(args.root, args.name)
    os.makedirs(ep, exist_ok=True)
    print("=" * 60)
    print("这一条存到 %s" % ep)
    if args.run:
        print("自动模式：录制期间执行  %s" % args.run)
    else:
        print("先按住机械臂上的【示教按钮】拖动到起始姿态（重力补偿模式）")
        print("然后回车开始，做完整套抓取动作，再回车结束")
    print("=" * 60, flush=True)
    if not args.no_wait and not args.run:
        input()

    procs = [
        subprocess.Popen([VISION_PY, os.path.join(HERE, "rec_cam.py"), "--out", ep,
                          "--seconds", str(args.seconds + 30), "--fps", str(args.fps)]),
        subprocess.Popen([ARM_PY, os.path.join(HERE, "rec_arm.py"), "--out", ep,
                          "--seconds", str(args.seconds + 30)]),
    ]
    time.sleep(1.5)  # 让录制进程先起来
    t0 = time.time()
    print(">>> 开始录制", flush=True)
    try:
        if args.run:
            rc = subprocess.call(args.run, shell=True)
            print(">>> 命令结束 rc=%s，录制再延 1.5s" % rc, flush=True)
            time.sleep(1.5)
        else:
            last = -1
            while time.time() - t0 < args.seconds:
                el = int(time.time() - t0)
                if el != last:
                    print("    %2ds / %ds" % (el, int(args.seconds)), flush=True)
                    last = el
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n>>> 提前结束")
    stop_all(procs)
    merge(ep, args.fps)
    print("总时长 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
