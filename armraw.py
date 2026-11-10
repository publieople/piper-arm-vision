#!/usr/bin/env python3
"""直接从 CAN 帧读真实关节角（0x2A5/2A6/2A7，大端 int32，0.001 度）。
SDK 的 get_joint_angles() 在控制器报 FAILED 后会读到过期值，此时用这个。
用法: python armraw.py            -> 打印真实关节角
"""
import can, struct, time

def read_true(timeout=1.0):
    bus = can.interface.Bus(channel="can0", interface="socketcan")
    last = {}
    t0 = time.time()
    try:
        while time.time() - t0 < timeout and len(last) < 3:
            m = bus.recv(0.3)
            if m is None:
                continue
            if m.arbitration_id in (0x2A5, 0x2A6, 0x2A7) and len(m.data) == 8:
                a, b = struct.unpack(">ii", bytes(m.data))
                last[m.arbitration_id] = (a * 0.001, b * 0.001)
    finally:
        bus.shutdown()
    if len(last) < 3:
        return None
    return [last[0x2A5][0], last[0x2A5][1], last[0x2A6][0], last[0x2A6][1],
            last[0x2A7][0], last[0x2A7][1]]

if __name__ == "__main__":
    q = read_true(1.5)
    print("TRUE_JOINTS", None if q is None else [round(v, 2) for v in q])
