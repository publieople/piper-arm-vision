# π0.5 部署到 GPU 服务器（itbk）

2026-11-10 记录。目标：把 π0.5（VLA，3.35 B 参数）跑在 `server-itbk`（4×A10 23 GB）上，
本机只做 websocket 客户端。**本地 8 GB 显存放不下**，原因见 [π0.5评估.md](π0.5评估.md)。

## 1. 结论（先看这个）

| 项目 | 结果 |
|---|---|
| 权重 | `pi05_libero` 12.44 GB（6 个 ocdbt 分片 + norm stats），完整 |
| 环境 | `/data/openpi/.venv` 7.7 GB，必须用**系统 python3.11**（uv 自带的 3.11.17 缺 `_decimal`，evdev 编不过） |
| 加载 | 374 s（12.4 GB 从 /data 冷读 + jax 初始化） |
| 推理 | 首帧 65 s（JIT 编译），**稳态 0.171 s/次**，动作块 `(10, 7)` |
| 显存 | bytes_in_use **6.94 GB**、peak 7.16 GB（A10 22 GB 放得下） |
| 坑 | 这台机 `available` 常只剩 5~10 GB（别人占着 120/125 GB），**加载时 12 GB 会换页到卡死** |

## 2. 装环境

```bash
rsync -a --exclude=.venv --exclude=.git --exclude=third_party \
  ~/openpi/ itbk:/data/openpi/

ssh itbk
cd /data/openpi
env UV_CACHE_DIR=/data/uv-cache uv sync --frozen --python /usr/bin/python3.11
```

`--python /usr/bin/python3.11` 是关键：uv 自己下载的 CPython 缺 `_decimal`，evdev 构建会失败
（`Failed to build evdev==1.9.2` / `No module named '_decimal'`）。

## 3. 下权重

`gs://openpi-assets/checkpoints/pi05_libero` → `/data/openpi-cache/pi05_libero`

- 服务器直连 GCS **13.9 MB/s**（本机走代理只有 0.85 MB/s，差 16 倍）
- 结构：`params/{_METADATA,_sharding,array_metadatas,d,manifest.ocdbt,ocdbt.process_0}` +
  `assets/physical-intelligence/libero/norm_stats.json`
- 校验：`du -sb` 应为 **12,439,085,481** 字节

## 4. 冒烟推理

```bash
cd /data/openpi
OPENPI_DATA_HOME=/data/openpi-cache CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python pi05_smoke.py
```

```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies.libero_policy import make_libero_example

cfg = _config.get_config("pi05_libero")
policy = _policy_config.create_trained_policy(cfg, "/data/openpi-cache/pi05_libero")
out = policy.infer(make_libero_example())   # -> {"actions": (10, 7)}
```

观测键：`observation/state`、`observation/image`、`observation/wrist_image`、`prompt`。

## 5. 起服务 + 本机连

```bash
# 服务器（注意 --port 是父级参数，必须放在 policy:checkpoint 之前，否则 tyro 报 Unrecognized options）
cd /data/openpi
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
nohup .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi05_libero --policy.dir=/data/openpi-cache/pi05_libero \
  > serve.log 2>&1 &

# 本机
ssh -N -L 8000:127.0.0.1:8000 itbk &
uv run --no-project --with websockets --with numpy python pi05/client_test.py
```

客户端代码在 `pi05/openpi_client/`（从 openpi 仓库拷出来的精简版，msgpack + websockets）。

## 6. 别再犯

- **加载前先看内存**：`free -g` 的 `available` 要 ≥ 20 GB，否则 12.4 GB 权重会把机器换页到失联
  （2026-11-10 就发生过一次，整台 itbk ping 不通、SSH 连不上）。
- 用完就 kill：`pkill -f serve_policy.py`。服务端 7 GB 显存 + 十几 GB 常驻内存，不适合长期占着。
- 只有一台空闲卡时用 `CUDA_VISIBLE_DEVICES=<n>` 指定，别让它把 4 张卡都占上。
- 当前 `pi05_libero` 是**仿真域（LIBERO 基准）**的策略，只能验证「部署 + 延迟」；
  要用在自己的机械臂上，必须拿本机采的示范做微调（先把 `act/` 那条线跑顺）。
