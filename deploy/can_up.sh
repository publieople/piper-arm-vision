#!/usr/bin/env bash
# 把 PiPER 的 USB-CAN 模块（candleLight / gs_usb）拉成 can0 @ 1 Mbps
# 插拔 CAN 模块或重启后 can0 会变回 DOWN，重新跑一次本脚本即可。
set -e
if ! ip link show can0 >/dev/null 2>&1; then
  echo "✗ 没有找到 can0：USB-CAN 模块没插好？(lsusb 应为 1d50:606f)"; exit 1
fi
ip link set can0 down 2>/dev/null || true
ip link set can0 type can bitrate 1000000
ip link set can0 up
ip -details link show can0 | head -4
echo "✓ can0 已 up @1Mbps（要 root，本脚本需 sudo 运行）"
