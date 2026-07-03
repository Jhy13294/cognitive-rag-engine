# 学习笔记索引：企业级 RAG 知识库

本文件作为学习笔记入口保留。详细内容已经拆分到三份更适合长期维护的高级笔记中：

- [技术选型思考](./tech-selection.md)：解释为什么选择当前模型、框架、协议、向量库、检索策略、服务协议和缓存设计。
- [系统架构与数据流图](./architecture.md)：说明入库、查询、混合检索、父子分块、上下文装填、Redis 缓存和评估链路。
- [核心踩坑记录](./dev-log-crashing.md)：沉淀指标污染、上下文装填边界、假异步、假流式、缓存失效、FakeRedis、live Redis、事件循环绑定、镜像构建、自包含性、liveness/readiness 健康门禁、扫描 PDF 入库诚实性和冻结基线行尾假阳性等问题。

## 推荐阅读顺序

1. 先读 [系统架构与数据流图](./architecture.md)，建立整体链路视角。
2. 再读 [技术选型思考](./tech-selection.md)，理解各组件为什么这样选。
3. 最后读 [核心踩坑记录](./dev-log-crashing.md)，熟悉后续开发最容易踩到的边界。

## 最近更新重点

- 生成质量评估已形成双轨：live 轨使用 DeepSeek `deepseek-v4-pro` 判官；Answer Relevance 的 embedding 默认走本地 fastembed `BAAI/bge-small-en-v1.5`（384 维，无 key、无配额、可完全离线），Gemini `gemini-embedding-001` 作为可选 provider。CI 轨只回放冻结 verdict，不联网、不发送原文。
- live capture 将答案生成温度与判官温度分别钉 0 并分别记录；每条正样本重复五次，case 门禁使用 median，噪声 margin 使用 MAD。mean/stddev 继续作为诊断信息，不再决定门禁。
- Faithfulness 的排障已深入到 `生成答案 -> statement 抽取 -> NLI verdict -> 聚合` 中间态。当前同时防两类伪零：联合要求被过度拆分，以及单句 `is relevant because` 被剥掉 NLI 所需的问题语境。
- Faithfulness 单独消费 `[n] filename: content` 格式的上下文以验证引用归属；Context Precision/Recall 继续消费纯正文。付费定点验证中，6 条 exact-name 正样本 Faithfulness median 全为 1.0，构造的引错源反例仍被判低，证明该格式增强了引用验证而非放宽评分。
- 生成质量 gate 只保留 Faithfulness、Answer Relevance 与 negative abstention/fabrication。Context Precision/Recall 仍完整计算并进入 case、capability 和总报告，但标记为 reported-only，因为它们不消费生成答案、只反映冻结的 hash 检索装置。
- 生成提示按问题形状分桶校准到“答案 + 最小支撑短语”的中间量：裸答案会让判官反推问题信息不足、过度脚手架又会漂离原问，两端都会拉低 Answer Relevance。这一路用全量真打逐版收敛，过程中实测到两个 gated 维度可能在同一答案上对冲（场景条件 vs Faithfulness）。
- verdict fixture 已升级为 `ragas-verdicts-v5`，强制 pin judge model、Ragas 版本、生成 prompt、statement prompt、Faithfulness context format 与 gating scope。协议漂移会使旧基线失效。
- Answer Relevance 使用 cosine，相似度可能因单位向量浮点尾差略超 1。两条 embedding 路都先做 L2 归一化，live capture 再以 `1e-6` 容差 clamp，NaN/inf 和真实越界仍致命；fixture replay 继续严格要求 `[0,1]`。
- 已冻结首份通过门禁的正式 verdict fixture：16 条正样本的 Faithfulness 与 Answer Relevance median 全部达标、负样本全部弃答，并由独立第二份 capture 确认稳定，离线 replay 自证门禁通过。约 20 条人工样本上的四维分数仍是带方差的判官估计，不是 production 真值。
- 服务已经能一键起全栈：多阶段构建的 serving 镜像 + compose 编排（向量库、缓存、元数据库、应用，服务名互联、healthcheck + 就绪依赖）+ 幂等初始化 + 无密钥也能起栈跑入库与检索的冒烟路径。打包依赖清单单独精简并钉死版本、删掉不 import 的重依赖，运行期资源在构建期固化以支持断网起容器。
- 健康检查拆成语义相反的两个探针：存活探针恒 200 只表示进程活着（容器编排用它，避免后端抖动误重启），就绪探针 fail-closed 按配置探启用的后端、任一必需后端不通即 503 并点名（交编排层做流量准入）。探针并发、有超时上界、只读探向量库、不泄敏感信息；结构性配置启动即校验、缺生成密钥降级为就绪非阻断依赖。验证 fail-closed 只认真断后端，mock 不算数。
- 持续集成新增两道无密钥离线门禁：一道清空外部服务环境变量后跑完整测试套件，把运行/失败/错误/跳过账目连同每类跳过原因的条数钉成精确常量，防“静默跳过伪装全绿”；一道以 `--no-cache` 实跑 dense、hybrid、parent-child 检索评估，把 MRR@3 与冻结基线做六位小数精确比对。Linux CI 与本地跑出的账目和分数逐位一致。

## 维护约定

- 公开文档以能力名和工程概念组织内容。
- 新增架构决策写入 `tech-selection.md`。
- 新增数据流、模块边界和系统图写入 `architecture.md`。
- 新增排查记录、事故复盘和评估口径修正写入 `dev-log-crashing.md`。
