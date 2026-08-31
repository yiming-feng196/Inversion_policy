# Frozen Flow Policy — Latest Retrieval/QKV Release

这是当前实验代码的精简发布版，只保留最后采用的 Flow-condition QKV
inverse-memory 实现。它是 MomentVLA/RoboVerse 项目的补丁层，不包含
IsaacSim、checkpoint、数据集或其他大文件。

## 当前实现

推理时使用统一的 frozen Flow condition embedding：

- `Q`：当前 observation 经过 Flow policy observation encoder 得到的
  `global_cond`；
- `K`：专家库中 action chunk 对应的同一 Flow condition embedding；
- `V`：专家 action chunk 通过 frozen Flow 反演到 `t*=0.8` 的 inverse state；
- attention：temperature-scaled dot-product attention，并支持 Top-M；
- forward：将 attention 融合后的 source 与当前 observation condition 一起送入
  frozen Flow forward。

新版不在在线查询时运行 DINOv3；DINO/proprio 版本属于旧实验，不在本发布版中。

## 文件

`patches/roboverse_learn/il/policies/fm/` 下的两个文件覆盖到现有
MomentVLA 项目同名路径：

- `action_memory_qkv.py`：参数无关的 memory attention/QKV source fusion；
- `retrieval_sequence_locked.py`：memory 构建、Flow-condition QKV runner、
  rollout 入口和日志记录。

该补丁依赖原始 MomentVLA/RoboVerse 代码中的：

- `visual_phase_inversion_skill_bank.py`；
- `pickcube_phase_inversion_cross_episode_clustering.py`；
- `roboverse_learn.il.runners.default_runner.DefaultRunner`；
- 原有 frozen Flow checkpoint、normalizer 和数据集。

## 应用补丁

在 MomentVLA 项目根目录执行：

```bash
cp patches/roboverse_learn/il/policies/fm/action_memory_qkv.py \
   roboverse_learn/il/policies/fm/action_memory_qkv.py
cp patches/roboverse_learn/il/policies/fm/retrieval_sequence_locked.py \
   roboverse_learn/il/policies/fm/retrieval_sequence_locked.py
python -m py_compile \
   roboverse_learn/il/policies/fm/action_memory_qkv.py \
   roboverse_learn/il/policies/fm/retrieval_sequence_locked.py
```

新版 runner 方法名：

```text
flow_condition_qkv_inverse
flow_condition_qkv_adaptive_depth_inverse
```

主方法固定使用 `t*=0.8` 的专家 inverse source；adaptive-depth 仅作为可选
诊断路径，不能与主结果混用。

## 复现实验注意事项

- 保持 frozen checkpoint、normalization、action horizon、observation history、
  ODE solver 和 `DefaultRunner` 不变；
- 评测数据和 checkpoint 不提交到 Git；
- 建议使用 `CUDA_VISIBLE_DEVICES` 明确绑定 GPU，并为每次运行保存完整命令；
- 当前版本仅表示代码整理完成，不代表某个未确认的远程 smoke test 已经成功。

