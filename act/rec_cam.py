#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录制腕部相机彩色流（给 ACT 采数据用）。

用 /home/po/vision-env/bin/python 跑（有 pyorbbecsdk）。

    python rec_cam.py --out act_data/ep_0000 --seconds 40 --fps 20

产出：<out>/cam.mp4（H.264 不行，用 mp4v）+ <out>/cam_ts.csv（t, frame_idx）
"""
import argparse, csv, os, sys, time
import numpy as np
import cv2
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def to_bgr(cf):
    fmt, h, w = cf.get_format(), cf.get_height(), cf.get_width()
    data = np.frombuffer(cf.get_data(), dtype=np.uint8)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.BGRA:
        return data.reshape(h, w, 4)[:, :, :3].copy()
    raise RuntimeError("unsupported fmt %s" % fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--rotate", type=int, default=180)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipeline = Pipeline()
    cfg = Config()
    cp = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    cfg.enable_stream(cp)
    dp = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    cfg.enable_stream(dp)          # 深度不录，但开着一起来帧更快、也更稳
    pipeline.start(cfg)

    t0 = time.time()
    while time.time() - t0 < 12:
        f = pipeline.wait_for_frames(1000)
        if f is not None and f.get_color_frame() is not None:
            break
    else:
        print("warn: 12s 内彩色流没出帧", flush=True)

    writer = None
    rows = []
    t_start = time.time()
    deadline = t_start + args.seconds
    next_dt = 1.0 / max(1.0, args.fps)
    idx = 0
    try:
      while time.time() < deadline:
          f = pipeline.wait_for_frames(200)
          if f is None:
              continue
          cf = f.get_color_frame()
          if cf is None:
              continue
          bgr = to_bgr(cf)
          if args.rotate == 180:
              bgr = cv2.rotate(bgr, cv2.ROTATE_180)
          if writer is None:
              h, w = bgr.shape[:2]
              writer = cv2.VideoWriter(os.path.join(args.out, "cam.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
              print("记录 %dx%d @ %.0f fps" % (w, h, args.fps), flush=True)
          now = time.time()
          # 按目标帧率抽帧（相机本身 30fps）
          if idx > 0 and now - t_start < idx * next_dt:
              continue
          writer.write(bgr)
          rows.append((now - t_start, idx))
          idx += 1
    except KeyboardInterrupt:
        print("rec_cam: 收到中断，收尾", flush=True)
    if writer is not None:
        writer.release()
    with open(os.path.join(args.out, "cam_ts.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "frame"])
        w.writerows(rows)
    pipeline.stop()
    print("cam done: %d frames / %.1fs" % (len(rows), time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
