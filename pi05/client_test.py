#!/usr/bin/env python3
"""pi05 客户端冒烟测试：连 itbk 上的 websocket policy 服务，发一帧假观测，打印动作块与耗时。

用法（先建隧道）：ssh -N -L 8000:127.0.0.1:8000 itbk
    uv run --no-project --with websockets --with numpy python client_test.py [host] [port]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/po/vision/pi05")
from openpi_client import websocket_client_policy as _ws  # noqa: E402

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

pol = _ws.WebsocketClientPolicy(host, port)
print("连接成功，服务端 metadata：", pol.get_server_metadata(), flush=True)

img = np.zeros((224, 224, 3), dtype=np.uint8)
obs = {
    "observation/state": np.zeros(8, dtype=np.float32),
    "observation/image": img,
    "observation/wrist_image": img,
    "prompt": "pick up the bottle",
}
t = time.time()
r1 = pol.infer(obs)
print("首次推理 %.2fs（含 JIT，服务端已编译则应很快）" % (time.time() - t), flush=True)
for i in range(5):
    t = time.time()
    r = pol.infer(obs)
    a = np.asarray(r["actions"])
    print("第 %d 次 %.3fs  actions %s" % (i + 1, time.time() - t, a.shape), flush=True)
print("\n动作块前 2 行：\n", np.asarray(r1["actions"])[:2])
