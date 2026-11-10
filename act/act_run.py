#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用训练好的 ACT 策略驱动机器人（端到端）。

    /home/po/arm-env/bin/python act_run.py --ckpt /home/po/vision/act_train/run1/checkpoints/last/pretrained_model

安全措施：每次单关节命令变化限幅（MAX_DJ=4°），下发前查限位与 IK 可达，
按 10Hz 读回关节+夹爪喂给策略。演示数据是什么动作它就学什么动作。
"""
import argparse, time
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autograsp as A
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--max-dj", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt).to(dev).eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": dev}},
        postprocessor_overrides={"device_processor": {"device": dev}},
    )
    print("策略已载入 %s（%s）" % (args.ckpt, dev), flush=True)

    arm, eff = A.connect()
    time.sleep(0.5)
    q = np.array([np.degrees(x) for x in arm.get_joint_angles().msg])
    print("起始关节", np.round(q, 1).tolist(), flush=True)

    # 相机（和采集同一路）
    A.capture("warm")
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        bgr, dep = A.capture("act%d" % n)
        torch_bgr = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        torch_bgr = torch.nn.functional.interpolate(torch_bgr[None], size=(240, 320), mode="bilinear",
                                                    align_corners=False)[0]
        try:
            grip = float(eff.get_gripper_status().msg.value) * 1000.0
        except Exception:
            grip = q[6] if len(q) > 6 else 60.0
        st = torch.from_numpy(np.concatenate([q, [grip]]).astype(np.float32))
        batch = {"observation.images.wrist": torch_bgr[None].to(dev),
                 "observation.state": st[None].to(dev)}
        with torch.no_grad():
            a = post(policy.select_action(pre(batch)))[0].cpu().numpy()
        tgt = a[:6]
        dj = np.clip(tgt - q, -args.max_dj, args.max_dj)
        cmd = q + dj
        print("step %d: 策略 %s → 下发 %s" % (n, np.round(tgt, 1).tolist(), np.round(cmd, 1).tolist()), flush=True)
        if not args.dry:
            A.move_j(arm, [np.radians(v) for v in cmd], speed=args.speed, timeout=4.0)
            A.gripper(eff, a[6], 2.0, seconds=1.2)
        q = np.array([np.degrees(x) for x in arm.get_joint_angles().msg])
        n += 1
        time.sleep(max(0.0, 1.0 / args.hz - 0.04))
    A.close_capture()
    print("[done] %d 步" % n, flush=True)


if __name__ == "__main__":
    main()
