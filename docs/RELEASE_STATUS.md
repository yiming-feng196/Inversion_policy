# Release status

日期：2026-08-31

## 保留

- Flow-condition QKV：当前 observation condition 作为 Q；
- 专家 Flow condition 作为 K；
- `x0.8` inverse state 作为 V；
- Top-M attention、temperature 和 attention logging；
- 与原始 frozen Flow/DefaultRunner 的集成入口。

## 排除

- 旧版 DINOv3 image-Q/K + proprio rerank QKV；
- 旧版单纯 Top-1/Top-K retrieval 作为最终实现；
- 旧版 candidate gate、prior reset、partial-inversion 和 adapter 实验；
- checkpoint、数据集、视频、缓存、临时 patch 和实验输出。

## 验证

- 本机 Python syntax compilation：通过；
- 远程新版文件上传与 smoke test：待 GitHub 整理后、远程服务器恢复可访问时重新确认。

