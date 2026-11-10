#!/usr/bin/env python3
"""抓一对对齐的 RGB-D 并落盘（自动按安装方向旋转）。
用法:
  python capture.py --rotate 180 --tag name      # 单次(进程启动约 0.9s)
  python capture.py --rotate 180 --serve         # 常驻: stdin 读 tag, stdout 回 OK(启动一次 0.9s, 之后每帧约 0.3s)
需要 PYTHONPATH=/tmp/pyorbbecsdk_v1/install/lib
"""
import os, sys, time, argparse
import numpy as np
import cv2
from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat,
                         AlignFilter, OBStreamType)

OUT = "/home/po/vision/out"


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
    raise RuntimeError(f"unsupported {fmt}")


def grab(pipeline, align, rotate, tries=8, min_valid=0.02):
    """取一帧。深度在近端(~<200mm)常常整帧为 0, 等下去也不会有回波,
    所以只等 8 帧(~0.27s)就返回最好的一帧; 检测本身不依赖深度。"""
    best, best_n = None, -1
    for _ in range(tries):
        # 等 1000ms: 相机空闲一阵后第一帧常要 2~3s, 200ms 会一路拿到 None 然后报 no frames
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            continue
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if cf is None or df is None:
            continue
        color = to_bgr(cf)
        aligned = align.process(frames)
        adf = aligned.get_depth_frame() if aligned else df
        d = np.frombuffer(adf.get_data(), dtype=np.uint16).reshape(
            adf.get_height(), adf.get_width()).copy()
        if rotate == 180:
            color = cv2.rotate(color, cv2.ROTATE_180)
            d = cv2.rotate(d, cv2.ROTATE_180)
        elif rotate == 90:
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
            d = cv2.rotate(d, cv2.ROTATE_90_CLOCKWISE)
        n = int(np.count_nonzero(d))
        if n > best_n:
            best, best_n = (color, d, cf.get_format(), adf.get_depth_scale()), n
        if n >= min_valid * d.size:
            return color, d, cf.get_format(), adf.get_depth_scale()
    if best is not None:
        print("warn: depth valid %d px only (near-limit?)" % best_n)
        return best
    raise RuntimeError("no frames")


def open_pipeline():
    pipeline = Pipeline()
    cfg = Config()
    cp = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    cfg.enable_stream(cp)
    dp = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    cfg.enable_stream(dp)
    print("color:", cp.get_width(), "x", cp.get_height(), cp.get_fps(), "fps", cp.get_format(), flush=True)
    print("depth:", dp.get_width(), "x", dp.get_height(), dp.get_fps(), "fps", dp.get_format(), flush=True)
    pipeline.start(cfg)
    # 预热: 刚 start 后彩色流常要 3~5s 才出第一帧(深度立刻就有), 不预热会误报 no frames
    t0 = time.time()
    while time.time() - t0 < 12:
        f = pipeline.wait_for_frames(1000)
        if f is not None and f.get_color_frame() is not None:
            print("warmup %.1fs" % (time.time() - t0), flush=True)
            break
    else:
        print("warn: 彩色流 12s 内没出帧", flush=True)
    return pipeline, AlignFilter(OBStreamType.COLOR_STREAM)


def save(tag, color, depth, vis=False):
    cv2.imwrite(f"{OUT}/{tag}_color.png", color)
    np.save(f"{OUT}/{tag}_depth.npy", depth)
    if vis:
        v = np.clip(depth.astype(np.float32) / 3000.0 * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{OUT}/{tag}_depth_vis.png", cv2.applyColorMap(v, cv2.COLORMAP_JET))


def serve(pipeline, align, rotate):
    """常驻取图: 每读一行 tag 拍一张, 回 OK 或 ERR <原因>"""
    print("READY", flush=True)
    for line in sys.stdin:
        tag = line.strip()
        if tag in ("quit", "exit"):
            break
        if not tag:
            continue
        try:
            color, depth, fmt, scale = grab(pipeline, align, rotate)
            save(tag, color, depth)
            print("OK", flush=True)
        except Exception as e:
            print("ERR %s" % e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotate", type=int, default=180, choices=[0, 90, 180, 270])
    ap.add_argument("--tag", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serve", action="store_true", help="常驻模式: stdin 读 tag")
    ap.add_argument("--vis", action="store_true", help="额外存深度伪彩图")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pipeline, align = open_pipeline()
    if args.serve:
        try:
            serve(pipeline, align, args.rotate)
        finally:
            pipeline.stop()
        return

    try:
        color, depth, fmt, scale = grab(pipeline, align, args.rotate)
        p = pipeline.get_camera_param()
    finally:
        pipeline.stop()

    tag = args.tag or "rgbd"
    save(tag, color, depth, vis=args.vis)

    valid = depth[depth > 0].astype(np.float32) * scale
    print("color shape:", color.shape, "fmt:", fmt)
    print("depth shape:", depth.shape, "scale:", scale)
    if valid.size:
        print("valid px: %d (%.1f%%)  range %.0f~%.0f mm  median %.0f mm" % (
            valid.size, 100.0 * valid.size / depth.size, valid.min(), valid.max(), np.median(valid)))
        h, w = depth.shape
        c = depth[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30]
        c = c[c > 0]
        if c.size:
            print("center 60x60 median: %.0f mm" % (np.median(c) * scale))
    else:
        print("!! depth empty")
    print("color intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f (%dx%d)" % (
        p.rgb_intrinsic.fx, p.rgb_intrinsic.fy, p.rgb_intrinsic.cx, p.rgb_intrinsic.cy,
        p.rgb_intrinsic.width, p.rgb_intrinsic.height))
    print("depth intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f (%dx%d)" % (
        p.depth_intrinsic.fx, p.depth_intrinsic.fy, p.depth_intrinsic.cx, p.depth_intrinsic.cy,
        p.depth_intrinsic.width, p.depth_intrinsic.height))
    print("saved", tag)


if __name__ == "__main__":
    main()
