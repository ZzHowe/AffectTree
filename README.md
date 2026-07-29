# AffectTree

面向长视频情感计算的工具增强树状推理框架（研究原型，开发中）。

AffectTree 将长视频情感理解重构为对情感证据的主动"假设—验证"搜索：推理策略在多条并行
推理轨迹上按广度优先展开，通过时序导航工具定位候选片段，再调用主动视觉操作（人脸跟踪 /
人脸放大 / 肢体裁剪）与白盒情感符号分析工具，在 Valence–Arousal（VA）空间中构建视频级
情感弧线（Emotion Arc）与转折点证据链。

## 框架特性

- **树状情感推理**：节点维护状态 s = <q, H, O, R, A>（查询、时序情感假设、VA 观测、
  推理轨迹、动作历史），BFS 假设展开 + 工具执行 + 验证剪枝
- **三层工具集**：
  - 导航：TemporalZoom / TemporalJump / Sliding
  - 主动视觉操作：FaceTrack / FaceZoom / BodyCrop——高清人脸截图回灌 VLM 上下文
  - 符号情感分析：FaceAU / PoseGesture / Prosody / SceneTone / AffectSpot，
    全部基于传统 CV / 信号处理，输出可解释结构化符号
- **双通道证据**：图像通道（截图供 VLM 解读）与符号通道（数值线索）互验
- **训练**：Emo-RPO——层级奖励（分支 + 树级 + 情感证据质量：VA 轨迹 CCC + 转折点 F1@tIoU）
- **评测**：QA Acc / F1@tIoU / CCC / Evidence F1 / Paired Accuracy

## 目录结构

| 目录 | 说明 |
|---|---|
| `train/` | RL 训练框架（基于 verl / EasyR1），含树状工具调用 rollout 与奖励接入，用法见 `train/README.md` |
| `eval/` | 评测代码（并发树状推理引擎、baseline） |
| `affecttree/` | 树状情感搜索引擎、三层工具集、Emo-RPO 奖励、评测指标 |
| `configs/` | 默认配置 |
| `scripts/` | 推理 / 训练入口 |

## 当前状态

- `train/`、`eval/`：训练与评测基座，可运行
- `affecttree/`：早期骨架——工具接口、推理树、奖励函数与评测指标已定义；
  核心传统 CV 实现（CLM 特征点、Gabor 纹理、YIN 基频、KLT 跟踪等）以 TODO 标注，逐步填充中

## License

- `train/`：Apache License 2.0（基于 verl / EasyR1，见 `train/LICENSE`）
- 其余代码：MIT License
