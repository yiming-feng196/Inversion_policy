# Frozen Flow Inversion Latent Experiments

本仓库提供冻结 Flow Matching policy 下的 expert inversion latent 实验代码。代码依赖原始 MomentVLA/RoboVerse 工程、checkpoint、数据集和 latent cache；这些大文件不提交到 Git。

## 代码文件

| 文件 | 作用 |
|---|---|
| `flow_latent_predictor_common.py` | 公共底层接口：加载冻结 Flow、执行可微 forward sampling、读取 cache、处理 condition 和 action normalization，并检查模型完整性。 |
| `prepare_inverted_latent_dataset.py` | 将 expert action chunk 通过 Flow reverse integration 反演为 `z_star`，生成可断点续跑的 train/validation cache。 |
| `conditional_latent_relevance_test.py` | 在固定 condition 下比较 inversion、单次 Gaussian random 和 Random Oracle Best@100 的 action error。 |
| `native_cycle_vs_expert_distribution_audit.py` | 构造并审计 `z_native`、`z_cycle`、`z_expert`，计算 covariance spectrum、effective rank、whitening 和径向统计。 |
| `local_latent_region_test.py` | 以 `z_star` 为中心进行不同半径的局部 Gaussian sampling，并与全局 N(0,I) sampling 比较。 |
| `directional_latent_geometry_test.py` | 在相同 latent 扰动长度下比较 radial outward、radial inward 和随机 tangential direction。 |
| `latent_geometry_figures.py` | 计算冻结 Flow 的精确 Jacobian、`J_F^T J_F` 的切向谱，以及 radial/high-sensitivity 和 low/high-sensitivity 行为网格。 |
| `behavior_region_localizer.py` | 仅用 train episodes 构建 Behavior Region Bank，进行 50/200-step teacher ranking，训练 behavior-supervised region locator，并在 validation 上比较 retrieval、Gaussian 和 candidate oracle。 |
| `evaluate_behavior_region_policy.py` | 评估 BRL anchor、isotropic local、tangent local 与 tangent + norm projection 的 Stage-2 初始化几何。 |
| `evaluate_behavior_region_rollout.py` | 为现有 RoboVerse rollout wrapper 提供 paired seed/scenario 的 clean 与 OOD 闭环评测 harness。 |
| `evaluate_region_warmstart_speed.py` | 为 NFE sweep、intermediate-state warm-start 和同步 GPU wall-clock 测速提供统一 harness。 |
| `temporal_region_continuity_test.py` | 对同一 episode 的相邻 policy update 比较 current inversion、previous-latent reuse、Gaussian 和 observation retrieval，验证 behavior region 是否可追踪。 |

所有实验均使用冻结 checkpoint 和 200 步 forward/reverse Flow integration；Flow 参数不参与更新。

## 当前实验结论

我们研究了从 expert action 反演得到的 latent 是否包含与行为相关的结构。实验结果支持以下结论。

**1. Expert inversion latent 与具体行为高度相关。** 在 1915 个 validation conditions 上，inversion 的 executed action error 为 **0.0101**，单次 Gaussian random 为 **0.2629**。即使允许 Gaussian random 采样 100 次并事后根据真实 expert action 选择最优样本，Random Oracle Best@100 仍为 **0.0914**。因此 inversion 分别比 random 和 Oracle Best@100 低约 **26×** 和 **9×**，说明 `z_star` 不是任意先验样本，而是包含当前行为信息的条件 latent。

**2. Expert inversion latent 存在明显的二阶结构重分配。** 以 native latent covariance 为参照，expert/native eigenvalue ratio 在 leading modes 中显著大于 1，在 trailing modes 中显著小于 1，表明 expert variance 被集中到更少的 dominant directions。对应的谱熵 effective rank 从 native 的 **142.7** 降至 cycle 的 **137.9**，再降至 expert 的 **94.2**。

**3. 这种结构不只是 covariance 差异。** 使用 expert training covariance whitening 后，expert 的 `||z_w||^2` 标准差为 **37.86**，而 Gaussian 理论 `chi^2_144` 的标准差为 **16.97**；QQ 曲线在中心和 upper tail 均偏离 `y=x`。因此消除二阶 covariance 后，expert inversion latent 仍保留 higher-order non-Gaussian structure。

**4. Behavior-specific latent region 具有有限半径。** 以 `z_star` 为中心的局部 sampling 在 RMS 半径 sigma = 0.05、0.1、0.25、0.5 下的平均 action error 分别约为 **0.0132、0.0196、0.0472、0.1389**，均低于 global Gaussian 的 **0.2629**；当 sigma = 1.0 时 error 上升至约 **0.8944**。这支持“存在 behavior-specific latent region”，而不是只有一个孤立的特殊 latent point。

**5. 局部区域具有 radial asymmetry 和 directional anisotropy。** 在相同 latent 扰动长度下，RMS 半径 0.75 时，radial outward、radial inward 和 tangential perturbation 的平均 action error 约为 **2.178、0.189、0.378**。此外，冻结 Flow 在 `z_star` 处的 `J_F^T J_F` 显示不同切向方向具有显著不同的 behavioral sensitivity。因此局部 behavior region 既不是以 `z_star` 为中心的圆，也不是各向同性 Gaussian ball，而是一个具有方向性和径向不对称的区域。
<img width="3380" height="1404" alt="tangential_anisotropy_low_high" src="https://github.com/user-attachments/assets/ce31f482-5e63-4cd0-af7f-86def89fe692" />
<img width="1820" height="1612" alt="local_behavior_region_radial_high" src="https://github.com/user-attachments/assets/2959e0bc-0abf-4d42-8947-ad054d9de308" />
<img width="4760" height="1540" alt="global_latent_structure_main" src="https://github.com/user-attachments/assets/4a926005-ec49-4294-a63f-372eba98f25f" />

总体而言，实验形成了如下证据链：

```text
variance concentration
        →
higher-order non-Gaussian structure
        →
behavior-specific local geometry
```

这为后续的 `localize → locally sample/refine → decode` 方法提供了直接实验依据。

## 下一阶段 BRL 实验

`behavior_region_localizer.py` 的 bank 只包含 train-episode inversion latent。Locator 使用 observation feature 学习行为兼容区域的 soft teacher distribution；validation expert action 只用于离线计算 action error，不能进入 bank 或推理输入。默认先用 50-step Flow 生成训练标签，并用至少 5000 个 candidate pairs 与 200-step teacher 比较；Spearman 低于 0.90 时自动回退到 100 或 200 steps。

生成可用于 warm-start 的 cache 时，增加 `--save-intermediate-states`；该选项同时保存当前 proprio 和 `x_tau_025/050/075`。

```bash
python behavior_region_localizer.py \
  --repo /path/to/MomentVLA-main \
  --checkpoint /path/to/30.ckpt \
  --cache /path/to/inversion_cache \
  --output-dir /path/to/behavior_region_localization/stage1_locator

python evaluate_behavior_region_policy.py \
  --repo /path/to/MomentVLA-main \
  --checkpoint /path/to/30.ckpt \
  --cache /path/to/inversion_cache \
  --stage1-dir /path/to/behavior_region_localization/stage1_locator \
  --output-dir /path/to/behavior_region_localization/stage2_geometry
```

正式 Flow validation 固定 200 steps；rollout 和测速脚本需要通过 `--runner module:function` 接入现有环境 wrapper，因此本仓库不会虚构 simulator API。

## 基本运行方式

先生成 inversion cache：

```bash
python prepare_inverted_latent_dataset.py \
  --repo /path/to/MomentVLA-main \
  --checkpoint /path/to/30.ckpt \
  --zarr /path/to/dataset.zarr \
  --cache /path/to/inversion_cache
```

其余脚本使用相同的 `--repo`、`--checkpoint` 和 `--cache` 参数。依赖见 `requirements.txt`；数据集、checkpoint、cache 和生成结果应保存在仓库之外。
