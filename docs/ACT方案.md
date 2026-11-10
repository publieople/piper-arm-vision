# ACT 端到端抓取（本机 8GB 跑得动的那条路）

π0.5 在这台 4070 8GB 上不可行（3.35B 参数 / 参数树 12.5GB，实测 OOM），
所以主路线换成 **ACT**（Action Chunking Transformer，LeRobot 官方实现，resnet18 主干，
推理只要 0.2GB 显存）。π0.5 另外部署到 server-itbk 的 A10 上做对照。

## 1. 环境（已装好）

| 用途 | 路径 | 说明 |
|---|---|---|
| LeRobot / ACT 训练 | `/home/po/act-env` | python 3.11 + torch 2.10+cu128 + lerobot 0.4.4 |
| 机械臂 + 相机（采集/回放） | `/home/po/arm-env`、`/home/po/vision-env` | 采集脚本分两个进程跑，因为这俩 SDK 的 python 版本不同 |

**坑：本机 ffmpeg 是 9.0（libavutil.so.61），torchcodec 读不了视频** →
训练/回放一律加 `--dataset.video_backend=pyav`（av 15.1.0 已装）。

## 2. 采集（`vision/act/`）

```bash
# A. 人手示教：按住机械臂示教按钮拖到起始姿态 → 回车 → 拖动完成动作 → 回车
cd /home/po/vision
/home/po/act-env/bin/python act/collect.py --seconds 40

# B. 不用人：录脚本跑出来的动作（自动攒演示数据）
/home/po/act-env/bin/python act/collect.py \
  --run "/home/po/arm-env/bin/python /home/po/vision/bodygrasp.py"
```

一条 = `act_data/ep_XXXX/`：
- `cam.mp4` 640×480@20fps（腕部彩色，直连 pyorbbecsdk，`--rotate 180` 已摆正）
- `cam_ts.csv` / `arm.csv`：各自时间轴，关节 100Hz、夹爪 20Hz
- `ep.npz`：按相机帧时间插值后的 `t/q[6]/grip/fps`

## 3. 转 LeRobot 数据集

```bash
/home/po/act-env/bin/python act/to_lerobot.py --overwrite
# → act_data/lerobot  (observation.images.wrist 320×240, observation.state(7), action(7))
```

state = [J1..J6 度, 夹爪开度 mm]；action = 下一帧的 state。

## 4. 训练

```bash
cd /home/po/vision && ./act/train_act.sh 20000 8
# 等价于 python -m lerobot.scripts.lerobot_train --policy.type=act --dataset.video_backend=pyav ...
# 产物 act_train/run1/checkpoints/last/pretrained_model
```

实测（假数据 3 条 360 帧，320×240，batch 8）：**11.5 step/s，显存 0.23GB**。
真实数据建议 2 万步起，1 小时量级。

## 5. 上机跑

```bash
/home/po/arm-env/bin/python act/act_run.py \
  --ckpt /home/po/vision/act_train/run1/checkpoints/last/pretrained_model --seconds 30
```

10Hz 读关节+夹爪+图像喂策略 → 关节目标限幅 4°/步下发。`--dry` 只看不下发。

## 6. 已验证 / 未验证

- ✅ 数据集格式、训练、checkpoint、策略推理（`select_action`）全链路跑通（用合成数据）
- ✅ `act_run.py` 的 pre/post processor 与训练一致（`make_pre_post_processors`）
- ⬜ 真实演示数据采集（**需要机械臂在线**）
- ⬜ 真机回放效果
