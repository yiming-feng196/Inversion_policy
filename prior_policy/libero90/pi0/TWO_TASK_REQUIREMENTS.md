# π₀ × LIBERO-90：双任务起点先验实验需求

日期：2026-09-22。状态：**实验需求，未启动；第二任务的低基线条件待确认。**

本需求只覆盖 LIBERO-90，不使用 LIBERO-Goal 或 LIBERO-LONG。保持 π₀ 动作生成器冻结，比较标准高斯起点、EIP-Gaussian 与 EIP-Flow。不微调 π₀，不增加示范数据。本文档不修改 `active_scope.json`，不恢复已暂停的 π₀/DP 队列，也不占用正在运行的 FM 队列资源。

## 1. 任务选择及证据

任务编号沿用本仓库 `../tasks.json` 的 **0-based 官方 LIBERO-90 注册顺序**；启动时同时核对完整名称，不能只依赖编号。

| 优先级 | ID / key | 完整任务名 | 选择依据 |
|---|---|---|---|
| 1 | 35 / `t035` | `KITCHEN_SCENE7_open_the_microwave` | 当前 π₀ 已有低基线记录，延续已有弱基策略起点适配实验。 |
| 2 | 8 / `t008` | `KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it` | 包含抽屉接触操作和物体放置；与当前 FM-UNet/DiT 评测同任务，便于跨架构对照。π₀ 成功率尚未确认。 |

服务器已有两项公开示范文件：

```text
/data/jhr/LIBERO/datasets/libero_90/KITCHEN_SCENE7_open_the_microwave_demo.hdf5
/data/jhr/LIBERO/datasets/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo.hdf5
```

2026-09-22 只读核查：历史目录 `/data/jhr/pi0_prior_20260919/rollouts/microwave_s{0,1,2}/results.json` 中，高斯基线分别为 **1/50、1/50、2/50**。这些重复基线采用相同初始状态和噪声种子，不能合并成 150 次独立试验；差异也提示需要记录运行时非确定性。它们只用于选题，不能替代本轮协议一致的正式基线。旧先验数据采用 stride-10，不得当作本轮 stride-8 的结果。

尚未核实到任务 8 的可靠 π₀ 成功率，**不得填入猜测的 50%–60%，也不得借用同名 LIBERO-Goal/LONG 任务的成功率**。

### 先确认基线，再花时间反演

1. 通过输入、动作变换和解码器预检后，对两个任务分别运行 50 个固定初始状态的高斯基线；无需先构建反演数据集。
2. 优先希望第二任务落在约 20%–70% 的非饱和区间。若任务 8 基线高于 70% 或低于 20%，保留全部记录并汇报，再决定是否替换；不得静默扩展任务或按 EIP 的收益挑任务。
3. 任务 35 明确保留为弱基策略案例，低于该区间不自动淘汰。但近零结果必须先排除图像变换、语言指令、归一化、动作语义与环境配置问题。
4. 正式三方法对照开始前冻结任务清单。不能把基于基线筛选的两任务平均值描述成完整 LIBERO-90 平均。

公开来源：官方 [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 和 [OpenPI LIBERO 实现](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)。本需求不声称上述两任务具有官方逐任务 π₀ 成功率报告。

## 2. 基础策略与数据协议

| 项目 | 本轮要求 |
|---|---|
| 冻结 checkpoint | 现有 `pi0_libero`，服务器路径 `/data/jhr/openpi_cache/openpi-assets/checkpoints/pi0_libero`；记录权重和资产哈希 |
| 方法 | G：标准高斯；CG：EIP-Gaussian；CF：EIP-Flow |
| 数据划分 | 每任务 50 条示范，按现有 π₀ `split_episodes` 划为 35 train / 5 val / 10 test，seed **20260919** |
| 先验输入 | 当前观测的冻结 VLM prefix 特征与归一化机器人状态；不输入未来观测、专家动作或轨迹编号 |
| 动作归一化 | 保留 checkpoint 原有统计量和输入/输出变换，不重新拟合 |
| 条件特征标准化 | 仅由 train 计算；CG/CF 使用同一统计量 |
| 起点形状 | 完整 `50 × 32`；不能只学习实际机器人使用的 7 个通道 |
| 推理解码 | 固定 stock-equivalent Euler-10，执行前 10 个动作，再基于新观测重规划 |
| 环境 | 官方 `libero_90`，指令取任务注册表，10 步 settling，最多 400 步任务执行 |

本轮适配数据的划分不能证明与基础模型预训练数据无重叠，也不能据此声称未见任务或未见几何泛化。

## 3. 降采样：反演前进行 stride-8 窗口抽取

对长度为 L 的每条示范，按时间顺序选取起始索引 `{0, 8, 16, ...} ∪ {L−1}` 并去重。**先选窗口，再反演**，不是反演全部数据后再丢弃样本。

- 保留每个起点对应的完整连续 50 步 action chunk；不将 chunk 内的动作每隔 8 步抽取，不改变控制频率。
- 末尾不足 50 步时沿用 repeat-last padding，记录 `valid_steps`；统计真实动作误差时屏蔽填充部分。
- 反演和先验学习保留全部 32 个模型坐标；报告误差时区分完整坐标、50×7 有效动作通道及前 10×7 执行片段。这三者不能混称“真实执行维度”。
- CG/CF 共用逐样本完全一致的训练缓存，不分别挑选示范，不按成功率或反演误差事后筛除样本。
- 记录每条轨迹的窗口索引、各 split 窗口数、完整未降采样窗口数和实际保留比例。
- 旧缓存只有在窗口索引、权重、代码、求解器、精度、条件特征及归一化全部一致时才可逐样本复用；不能把 stride-10 缓存直接标成 stride-8。

## 4. 反演与数值预检

沿用已验证的 π₀ 设置：**FP32 RK4-1280**，明确传入 `--steps 1280 --stride 8 --dtype float32`，不要依赖 `extract.py` 的 128 步默认值。比较 RK4-640 与 RK4-1280 的起点变化，并记录已知高斯起点恢复、高阶动作重建、实际 Euler-10 解码重建误差。

π₀ 原生时间约定为 action t=0、noise t=1，反演沿 0→1 积分。模型权重沿用 helper 加载后的 BF16 舍入值，再提升到 FP32，包括激活计算；G/CG/CF 必须使用同一精度。不得将该实现不加说明地标为原生 BF16 推理性能。

必须执行现有预检：动作输入/输出变换 max-abs ≤1e-4、同精度原生采样器与封装采样器一致性、现有高阶 roundtrip 和半步/全步收敛门限。保存预检数值及失败记录；失败不能跳过或偷偷放宽阈值。RK4 的高阶逆不是部署 Euler-10 的精确逆，部署重建残差必须单独报告。

训练、验证反演完成即可训练先验；测试轨迹反演仅用于离线诊断，不能参与训练、标准化或 checkpoint 选择。反演耗时较长，不得为了占满双卡中断其他实验；未来启动时先确定资源分配。

## 5. 先验训练与闭环评测

- CG：条件对角高斯 NLL；CF：条件 Flow Matching 速度回归，采样 Euler-16。两者监督均为相同的完整反演起点。
- 保留现有预算：每模型 5,000 updates，batch 32，AdamW，lr 1e-4，warmup 200 + cosine，weight decay 1e-6，gradient clip 1。
- 先验训练 seeds 0/1/2，使用固定最终 checkpoint。不能根据测试成功率挑 epoch 或 seed。
- 每任务固定同一组 50 初态，environment seed 7；沿用由 `(noise_seed=0, initial_state, query)` 确定的随机数规则，所有方法在整个 rollout 的每个重规划时刻使用各自起点策略。
- G 评测一次 50 回合，CG/CF 各 3×50 回合：**每任务 350、两任务 700 回合**。两个任务基线确认阶段的 100 回合在协议不变时直接计入，不额外重复。
- 每任务报告 G 的成功数/50，CG/CF 三个训练种子各自成功数及均值±样本标准差，和相对 G 的百分点变化；不要把先验训练 seed 当成独立基础策略训练 seed。
- 保存全部逐回合结果、失败、初态、查询次数、动作和日志。异常中断单独标记，恢复时避免重复计数；不能删除负结果。

## 6. 效率报告

每任务报告：stride-8 保留窗口数/完整窗口数、反演预检耗时和批量反演总耗时、缓存大小、每种先验的训练时长及峰值显存。推理分别报告起点生成和端到端策略查询的 warm p50/p95 延迟、冷启动/JIT 时间及峰值显存。

注明 GPU、JAX/PyTorch/CUDA 版本、精度、batch size、CF 步数和先验运行设备；GPU 计时必须同步。跨 JAX/PyTorch 进程记录各 PID 和整卡显存，不能只用 Torch allocator 数字代表总占用。不预先承诺 stride-8 带来 8 倍加速；若没有 full-stride 实测，仅报告保留比例和实际耗时。

## 7. 实施前必须处理的仓库依赖

当前 `pi0/` 是实验快照，不是独立可安装包。以下事项属于实现验收，本文档没有替代它们：

1. `runtime.py` 依赖外部 `pi0_flow_ops.py`（`PI0_HELPERS`），需要纳入受版本控制的依赖或明确获取方式与哈希。
2. `prior_models.py` 的外部 `source_models` 导入应指向仓库内实现；固定并验证所用 MomentVLA/OpenPI 代码版本，路径应可配置。
3. 补齐 JAX/Flax/OpenPI/openpi-client/LIBERO/MuJoCo 等实际可用版本及环境复现说明，不能仅依赖根目录通用 requirements。
4. 使用现有 `tasks.json` 的 `t035`、`t008`；构建独立、显式启动的 π₀ 双任务入口，不运行会扩展成全套任务的旧调度器。
5. 为 stride-8+末窗口去重、padding、split 不重叠、冻结参数、解码器一致性和断点续跑编写/复用测试。
6. 输出独立实验目录，保存 checkpoint/数据/代码哈希、完整参数、split、缓存来源和依赖版本。不得覆盖历史 π₀ 结果或当前 FM 实验。

验收产物：两任务配置与基线确认记录、数值预检、两份 stride-8 缓存清单、12 个先验最终 checkpoint、700 回合完整结果、成功率汇总及效率报告。如第二任务未满足基线选择要求，应明确交付为“待定任务”，不能假称两个低成功率任务已确认。
