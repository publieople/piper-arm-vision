#!/home/po/arm-env/bin/python
"""armctl.py — PiPER 机械臂统一入口(所有已实现功能)

  /home/po/arm-env/bin/python /home/po/vision/armctl.py status
  /home/po/arm-env/bin/python /home/po/vision/armctl.py home   [--speed 20] [--quick] [--force]
  /home/po/arm-env/bin/python /home/po/vision/armctl.py box     [--release]
  /home/po/arm-env/bin/python /home/po/vision/armctl.py bottle  [--lift --release]
  /home/po/arm-env/bin/python /home/po/vision/armctl.py screw   [--dry|--lift-only|--release|--force]
  /home/po/arm-env/bin/python /home/po/vision/armctl.py screw-top [--dry|--lift|--release] [--record]
  /home/po/arm-env/bin/python /home/po/vision/armctl.py cap     [--dry|--keep|--place X Y]

子命令都是转调既有脚本, 参数原样透传:
  box    -> autograsp.py    (方盒, 含角度伺服/螺旋搜索)
  bottle -> bottleside.py   (饮料瓶, 侧夹)
  screw  -> screwgrasp.py   (3D打印螺丝, 正上方跨轴夹)
  cap    -> capgrasp.py     (瓶盖: 深度定位 -> 倾斜下降 -> 闭合 -> 可搬运)
"""
import os, subprocess, sys

V = "/home/po/vision"
PY = "/home/po/arm-env/bin/python"     # 控制(pyAgxArm)
VPY = "/home/po/vision-env/bin/python"  # 视觉(pyorbbecsdk)

MAP = {
    "box":       (PY, f"{V}/autograsp.py",  []),
    "bottle":    (PY, f"{V}/bottleside.py", []),
    "screw":     (PY, f"{V}/screwgrasp.py", []),
    "screw-top": (PY, f"{V}/screwgrasp.py", ["--top"]),
    "cap":       (PY, f"{V}/capgrasp.py", []),
    "body":      (PY, f"{V}/bodygrasp.py", []),
}

def sh(cmd):
    print("$", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in ("status", "st"):
        return sh([PY, "/home/po/arm/arm.py", "status"])
    if cmd in ("home", "zero"):
        force = "--force" in rest
        quick = "--quick" in rest
        speed = 20
        if "--speed" in rest:
            speed = int(float(rest[rest.index("--speed") + 1]))
        return sh([PY, "-c", "import sys;sys.path.insert(0,'%s');"
                   "import autograsp as A;arm,e=A.connect();"
                   "ok=A.goto_home(arm, speed=%d, force=%s, precheck=%s);"
                   "arm.disconnect();sys.exit(0 if ok else 1)"
                   % (V, speed, force, (not quick))])
    if cmd in ("limits", "l"):
        return sh([PY, "/home/po/arm/arm.py", "limits"])
    if cmd == "look":            # 拍一张当前视角
        return sh([VPY, f"{V}/capture.py", "--rotate", "180", "--tag", "look"])
    if cmd in MAP:
        py, script, prefix = MAP[cmd]
        return sh([py, script, *prefix, *rest])
    print("未知子命令:", cmd); print(__doc__); return 1

if __name__ == "__main__":
    sys.exit(main())
