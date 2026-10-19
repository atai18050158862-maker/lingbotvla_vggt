# LingBot-VLA 2.0 + VGGT：多视角时空几何关系蒸馏

本仓库基于 [LingBot-VLA 2.0](https://github.com/robbyant/lingbot-vla-v2)。原版用两个教师（LingBot-Depth、DINO-Video）做 query 蒸馏，本仓库在此基础上加入冻结的 [VGGT](https://github.com/facebookresearch/vggt) 作为第三个教师，为 VLA 提供**跨视角一致、跨时间对应**的几何监督。新增模块只在训练时使用，推理计算图与原版完全一致。

## 方法概述

| 组件 | 说明 |
|---|---|
| VGGT 教师 | 冻结的 VGGT-1B，只使用 aggregator（交替进行帧内注意力和全局注意力的主干）。每个样本把 `[top_t, 左腕_t, 右腕_t, top_{t+T}, 左腕_{t+T}, 右腕_{t+T}]` 共 6 帧一次性联合输入，top_t 作为参考坐标系；取最后一层 2048 维的 patch 特征，每个视角池化到 8×8。缺失的相机不参与 VGGT 前向。 |
| 学生头 | 两个 resampler 头：当前头读取三个视角的 192 个图像 token 和当前 query `Q_t`，未来头读取同样的图像 token 和未来 query `Q_{t+T}`。复用原有的 query，prefix 长度不变；推理时不参与计算。 |
| 关系蒸馏损失 | 比较学生与教师特征的**均值归一化 pairwise L2 距离矩阵**（包含跨视角的 token 对），因此两边的特征空间和维度不需要对齐。时间项匹配从 t 到 t+T 的关系变化 `(R_{t+T} − R_t) / (|R_{t+T}| + |R_t|)`。 |

总损失：

```
L = L_action + L_depth + L_video + λ_rel · L_relation + λ_tem · L_temporal
```

默认 `λ_rel = 0.03`、`λ_tem = 0.1`。同一套关系损失也可以直接作用在原有的 LingBot-Depth 目标上，用于消融实验。

## 主要改动

**新增文件**

| 文件 | 作用 |
|---|---|
| `lingbotvla/models/vla/vision_models/vggt/` | VGGT aggregator 与 layers 源码（附 VGGT License） |
| `lingbotvla/models/vla/vision_models/vggt_teacher.py` | VGGT 教师：加载权重，对多视角的当前帧和未来帧做联合前向，并池化 patch 特征 |
| `lingbotvla/models/vla/vision_models/align_heads/vggt_align.py` | 学生端的 VGGT 对齐头（仅训练时使用） |
| `lingbotvla/models/vla/vision_models/align_heads/relation_loss.py` | 帧内关系损失与时间关系损失 |
| `configs/vla/robotwin/robotwin_vggt.yaml` | VGGT 版训练配置 |
| `configs/vla/robotwin/robotwin_abla.yaml` | 消融 baseline 配置（关闭 VGGT） |
| `scripts/prepare_robotwin_clean50.sh` | 一键把 RoboTwin 原始数据转换成 LeRobot v3.0 格式，并生成训练清单 |
| `assets/training_data/robotwin_clean50.txt` | 50 个任务的 clean_50 训练清单 |

**修改文件**

| 文件 | 改动 |
|---|---|
| `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` | 接入 VGGT 对齐头与关系损失，关系损失计入总 loss |
| `tasks/vla/train_lingbotvla.py` | 构建 VGGT 教师，每一步计算教师目标并传入模型 |
| `lingbotvla/models/module_utils.py` | post-training 从原版 checkpoint 起步时，允许 `vggt_align` 参数缺失，并对其初始化 |
| `train.sh` | 优先使用当前 conda 环境的 Python 和本仓库的代码；改用 `python -m torch.distributed.run` 启动 |
| `docs/config/lingbotvla_config_doc.md` | 补充 VGGT 与关系损失的配置说明 |

## 环境

```bash
bash tools/create_train_env.sh
conda activate lingbotvla
```

VGGT 部分只依赖 PyTorch，不需要额外安装。

## 权重

| 权重 | 来源 | 训练 | 评测 |
|---|---|:---:|:---:|
| LingBot-VLA 2.0 预训练 | `robbyant/lingbot-vla-v2-6b`（仓库根目录即 hf_ckpt，含 `depth/`、`dino_video/`） | ✓ | |
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` | ✓ | ✓ |
| MoGe-2 | `Ruicheng/moge-2-vitb-normal`（`model.pt`） | ✓ | |
| VGGT-1B | `facebook/VGGT-1B`（`model.pt`，只加载 aggregator） | ✓ | |
| 训练得到的模型 | `output/<实验名>/checkpoints/global_step_*/hf_ckpt` | | ✓ |

```bash
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir lingbot-vla
huggingface-cli download Ruicheng/moge-2-vitb-normal model.pt --local-dir /path/to/moge-2-vitb-normal
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct
huggingface-cli download facebook/VGGT-1B model.pt --local-dir /path/to/VGGT-1B
```

## 数据准备（RoboTwin 2.0，clean_50）

原始数据的目录结构为 `<DATA_ROOT>/<任务名>/aloha-agilex_clean_50/...`；解压时多出来的一层同名目录会被自动识别。

转换前需要按 [RoboTwin 官方说明](https://robotwin-platform.github.io/doc/usage/Pi0.html) 准备好 RoboTwin 仓库（commit `13c3c47`）和 `policy/pi0` 的 uv 环境：

```bash
cd /path/to/RoboTwin/policy/pi0 && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync
```

然后在训练环境里运行：

```bash
conda activate lingbotvla
DATA_ROOT=/path/to/robotwin/dataset \
LEROBOT_ROOT=/path/to/lerobot_output \
ROBOTWIN_ROOT=/path/to/RoboTwin \
bash scripts/prepare_robotwin_clean50.sh
```

脚本会依次完成：
1. 检查 50 个任务的数据是否完整；缺任务时直接停止，不会改动清单。加 `CHECK_ONLY=1` 只做检查。
2. 调用 RoboTwin 官方的 `process_data.py` 和 `generate.sh` 转换出 LeRobot 数据。
3. 自动升级为 LeRobot v3.0（训练环境里的 lerobot 0.4.2 只能读 v3.0）。原数据保留为 `*_old`，确认训练能正常读取后可以删除。
4. 写入 `assets/training_data/robotwin_clean50.txt`。

已完成的任务会被跳过，中途失败后重跑同一条命令即可。归一化统计沿用官方的 `assets/norm_stats/robotwin.json`。

## 训练

在 `configs/vla/robotwin/robotwin_vggt.yaml` 里填好各个权重路径。VGGT 相关的主要配置项：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `vggt.enable` | `true` | 是否启用 VGGT 蒸馏 |
| `vggt.ckpt_path` | — | VGGT-1B 的 `model.pt` |
| `vggt.views` | `[0, 1, 2]` | 参与 VGGT 的相机，下标对应 `data.cameras`；`[0]` 表示只用 top 视角 |
| `vggt.input_size` | `518` | VGGT 输入宽度 |
| `vggt.relation_loss_weight` | `0.03` | 帧内关系损失权重 |
| `vggt.temporal_relation_loss_weight` | `0.1` | 时间关系损失权重 |
| `vggt.cross_view_pairs` | `true` | 是否计入跨视角的 token 对 |

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin_vggt.yaml
```

- GPU 数由 `train.sh` 自动检测。不写 `global_batch_size` 时，程序会自动算成 `micro_batch_size × GPU 数 × 梯度累积步数`。
- 我们使用的设置：8 卡，每卡 12 个样本（总 batch 96），`enable_gradient_checkpointing: true`，共 50000 步（约 55 万帧数据上约 8.7 个 epoch）。
- 训练日志中的 `RelationAux_Loss` 是加权后的关系损失；tensorboard 中对应 `align/vggt_*` 等指标。

## 消融实验

| 组别 | 配置 | 相对完整方法的改动 |
|---|---|---|
| Baseline（原版 LingBot） | `robotwin_abla.yaml` | `vggt.enable: false` |
| + 关系损失（不用 VGGT） | 复制 baseline 配置 | `depth.relation_loss_weight: 0.03`，`depth.temporal_relation_loss_weight: 0.1` |
| + VGGT（单视角） | 复制 `robotwin_vggt.yaml` | `vggt.views: [0]` |
| + VGGT（三视角，完整方法） | `robotwin_vggt.yaml` | — |

除表中列出的改动外，各组的其余设置完全相同。

## 评测（RoboTwin 2.0 仿真）

先安装仿真环境（RoboTwin commit `13c3c47`）：

```bash
cd /path/to/RoboTwin && git submodule update --init --recursive
conda create -n RoboTwin python=3.10 -y && conda activate RoboTwin
bash script/_install.sh && bash script/_download_assets.sh
```

然后运行评测（在本仓库根目录执行）：

```bash
QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/output/<实验名>/checkpoints/global_step_50000/hf_ckpt \
  --eval_workdir /path/to/RoboTwin \
  --output_base /path/to/eval_results \
  --conda_sh $(conda info --base)/etc/profile.d/conda.sh \
  --inference_env lingbotvla --sim_env RoboTwin \
  --task_config demo_clean --num_tasks 50 --num_gpus <GPU数> --num_per_gpu 1
```

- 评测只需要训练得到的 hf_ckpt 和 Qwen3-VL；四个教师模型都不会被加载。
- 显存小于 32GB 的显卡（例如 RTX 4090）需要加上 `--use_bf16 True --use_fp32 False`。各组要使用相同的精度。
- 结果保存在 `<output_base>/<实验名>_<步数>k_<task_config>_<时间>/stats.txt`。

## 结果

| 方法 | Clean | Randomized |
|---|:---:|:---:|
| Baseline（原版 LingBot） | | |
| + 关系损失（不用 VGGT） | | |
| + VGGT（单视角） | | |
| + VGGT（三视角） | | |

## 致谢与许可

- 本仓库基于 [LingBot-VLA 2.0](https://github.com/robbyant/lingbot-vla-v2)（Apache-2.0）。
- `lingbotvla/models/vla/vision_models/vggt/` 下的代码来自 [VGGT](https://github.com/facebookresearch/vggt)，遵循 VGGT License（见该目录下的 `LICENSE.txt`）。
- 本仓库不包含任何模型权重和数据。
