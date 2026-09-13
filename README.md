# Expert-Inversion Prior Flow

我们在冻结的 Action Flow 前引入条件 Prior Flow，将标准高斯噪声映射到专家动作的反演 latent 分布，以保留基线已有的动作生成能力，并通过学习 source 分布使采样更贴近专家行为。首先，使用基线冻结的观测编码器得到统一条件 $c=E(o)$，在同一条件下通过 Action Flow 的反向数值积分得到 $z^\star\approx F_\theta^{-1}(A^\star\mid c)$，离线缓存 $(c,z^\star)$；随后训练与 Action Flow 采用相同 Conditional UNet1D 主干、但参数独立的速度网络 $v_\phi$：采样 $\epsilon\sim\mathcal N(0,I)$ 和 $t\sim U(0,1)$，构造 $z_t=(1-t)\epsilon+tz^\star$，最小化 $\mathcal L=\mathbb E\lVert v_\phi(z_t,t,c)-(z^\star-\epsilon)\rVert_2^2$。这种设计将条件分布学习与动作解码分开：训练仅使用缓存的条件和 latent、只更新 $\phi$，Action Flow 与观测编码器完全不参与反向传播；推理时从新的高斯噪声出发，对 Prior Flow 从 $t=0$ 积分到 $t=1$ 得到 $z_{\rm prior}=G_\phi(\epsilon,c)$，再由冻结的 $F_\theta(z_{\rm prior}\mid c)$ 生成动作，无需专家动作或反演缓存。Prior Flow 也可仅积分至 $t^*\in[0,1]$，以调节对原始高斯 source 的变换程度；$t^*=0$ 恢复高斯基线，$t^*=1$ 使用完整 Prior Flow。

![Expert-Inversion Prior Flow: offline inversion, conditional flow-matching training, and inference through a frozen Action Flow.](assets/prior_flow_architecture.svg)

[核心代码与使用说明](prior_flow/) · [架构图 SVG](assets/prior_flow_architecture.svg)
