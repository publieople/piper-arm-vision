#!/usr/bin/env bash
# 准备仿真用的 URDF + mesh（kin.py / simpath.py 用）
# 依赖：git、python3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/agx_arm_urdf"

if [ ! -d "$DEST" ]; then
  echo ">> clone agx_arm_urdf → $DEST"
  git clone --depth 1 https://github.com/agilexrobotics/agx_arm_urdf "$DEST"
else
  echo ">> 已存在 $DEST，跳过 clone"
fi

echo ">> 生成带绝对 mesh 路径的 URDF"
python3 - "$ROOT" "$DEST" <<'PY'
import pathlib, sys

root, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out_dir = dest / "piper" / "urdf"
out_dir.mkdir(parents=True, exist_ok=True)
for src in (root / "urdf").glob("*.urdf"):
    text = src.read_text().replace("__URDF_MESH_ROOT__", str(dest))
    (out_dir / src.name).write_text(text)
    print("   ", out_dir / src.name)
PY

echo ">> 完成。自检："
python3 - "$DEST" <<'PY'
import pathlib, sys
dest = pathlib.Path(sys.argv[1])
for name in ("piper_description.urdf", "piper_with_gripper.urdf"):
    p = dest / "piper" / "urdf" / name
    bad = [l for l in p.read_text().splitlines() if "__URDF_MESH_ROOT__" in l]
    print(f"    {name}: {'OK' if not bad else 'STILL TEMPLATED'}")
PY
echo ">> 之后运行：python tools/simselfcheck.py 或 kin.py 直接可用"
