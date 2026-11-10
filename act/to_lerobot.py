#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把采集到的 ep_XXXX 转成 LeRobot 数据集（给 ACT 训练用）。

    /home/po/act-env/bin/python to_lerobot.py --src act_data --repo local/piper_grasp

特征：observation.images.wrist（彩色，默认缩到 320x240）、observation.state(7)、action(7)
  state = [J1..J6(度), 夹爪开度(mm)]；action = 下一帧的 state（位置控制）
"""
import argparse, glob, os
import numpy as np
import cv2
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "grasp the bottle"


def load_ep(epdir, size):
    d = np.load(os.path.join(epdir, "ep.npz"))
    q, g = d["q"], d["grip"]
    cap = cv2.VideoCapture(os.path.join(epdir, "cam.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    n = min(len(frames), len(q))
    st = np.concatenate([q[:n], g[:n, None]], axis=1).astype(np.float32)
    act = np.concatenate([st[1:], st[-1:]], axis=0)
    return frames[:n], st, act, float(d["fps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/po/vision/act_data")
    ap.add_argument("--repo", default="local/piper_grasp")
    ap.add_argument("--root", default="/home/po/vision/act_data/lerobot")
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    W, H = [int(x) for x in args.size.lower().split("x")]

    eps = sorted(glob.glob(os.path.join(args.src, "ep_*")))
    eps = [e for e in eps if os.path.exists(os.path.join(e, "ep.npz"))]
    if not eps:
        raise SystemExit("没有找到 ep_*，先跑 collect.py")

    if args.overwrite and os.path.exists(args.root):
        import shutil
        shutil.rmtree(args.root)

    feats = {
        "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=args.repo, fps=20, features=feats,
                               root=args.root, use_videos=True,
                               image_writer_threads=4)
    for e in eps:
        frames, st, act, fps = load_ep(e, (W, H))
        for i in range(len(frames)):
            ds.add_frame({
                "observation.images.wrist": cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
                "observation.state": st[i],
                "action": act[i],
                "task": TASK,
            })
        ds.save_episode()
        print("episode %s: %d 帧 %dx%d" % (os.path.basename(e), len(frames), W, H), flush=True)
    ds.finalize()
    print("数据集在 %s（%d 条 / %d 帧）" % (args.root, ds.num_episodes, ds.num_frames))


if __name__ == "__main__":
    main()
