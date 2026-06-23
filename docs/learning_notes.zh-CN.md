# 学习笔记索引：企业级 RAG 知识库

本文件作为学习笔记入口保留。详细内容已经拆分到三份更适合长期维护的高级笔记中：

- [技术选型思考](./tech-selection.md)：解释为什么选择当前模型、框架、协议、向量库、检索策略、服务协议和缓存设计。
- [系统架构与数据流图](./architecture.md)：说明入库、查询、混合检索、父子分块、上下文装填、Redis 缓存和评估链路。
- [核心踩坑记录](./dev-log-crashing.md)：沉淀指标污染、上下文装填边界、假异步、假流式、缓存失效、FakeRedis、live Redis 和事件循环绑定等问题。

## 推荐阅读顺序

1. 先读 [系统架构与数据流图](./architecture.md)，建立整体链路视角。
2. 再读 [技术选型思考](./tech-selection.md)，理解各组件为什么这样选。
3. 最后读 [核心踩坑记录](./dev-log-crashing.md)，熟悉后续开发最容易踩到的边界。

## 最近更新重点

- 生成质量评估已形成双轨：live 轨使用 DeepSeek `deepseek-v4-pro` 判官和 Gemini `gemini-embedding-001@3072`，CI 轨只回放冻结 verdict，不联网、不发送原文。
- live capture 将答案生成温度与判官温度分别钉住并分别记录；每条正样本重复五次，case 门禁使用 median，噪声 margin 使用 MAD。mean/stddev 继续作为诊断信息，不再决定门禁。
- Faithfulness 的排障已深入到 `生成答案 -> statement 抽取 -> NLI verdict -> 聚合` 中间态。当前同时防两类伪零：联合要求被过度拆分，以及答案中的“文件名 + [n]”引用声明在 content-only context 中无法验证。
- Faithfulness 单独消费 `[n] filename: content` 格式的上下文；Context Precision/Recall 继续消费纯正文。付费定点验证中，6 条 exact-name 正样本 Faithfulness median 全为 1.0，构造的引错源反例仍被判低，证明该格式增强了引用验证而非放宽评分。
- 生成质量 gate 只保留 Faithfulness、Answer Relevance 与 negative abstention/fabrication。Context Precision/Recall 仍完整计算并进入 case、capability 和总报告，但标记为 reported-only，因为它们不消费生成答案、只反映冻结的 hash 检索装置。
- verdict fixture 已升级为 `ragas-verdicts-v4`，强制 pin judge model、Ragas 版本、statement prompt、Faithfulness context format 与 gating scope。协议漂移会使旧基线失效。
- Answer Relevance 的 Gemini cosine 可能因单位向量浮点尾差产生 `1.0000001`。live capture 只在 `1e-6` 容差内 clamp，NaN/inf 和真实越界仍致命；fixture replay 继续严格要求 `[0,1]`。
- Gemini Ragas adapter 四个同步/异步入口增加 L2 归一化作为纵深防御，但 clamp 才是最终数值边界。当前全量 250 项测试通过、8 项 gated skip；付费 candidate capture 与首份正式 fixture 仍待完成。

## 维护约定

- 公开文档使用能力名和工程概念，避免暴露项目内部施工标签。
- 新增架构决策写入 `tech-selection.md`。
- 新增数据流、模块边界和系统图写入 `architecture.md`。
- 新增排查记录、事故复盘和验收口径修正写入 `dev-log-crashing.md`。
