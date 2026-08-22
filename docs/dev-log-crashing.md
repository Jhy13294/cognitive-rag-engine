# 核心踩坑记录：我的面向企业 RAG 工程复盘

这份笔记记录我在项目开发中真正遇到的问题、修复过程和判断方式。它不是功能清单，也不是任务流水账。我更关心的是：为什么一个“看起来能跑”的实现仍可能污染指标、泄漏权限、失效缓存，或者让评估结论失真。

## 父块展开会虚抬检索指标

**现象**

我最初把 parent-child 展开后的 sources 直接交给检索评估器，MRR 和 recall 反而上升。继续检查后发现，兄弟子块折叠成同一个父块会压缩 top-k 窗口，原本排在窗口外的异源相关文档因此被顶了进来。指标变好来自列表形态变化，不是召回或排序能力提升。

**修复**

- 检索评估只使用原始 child ranked list。
- parent expansion 只在 top-k 之后服务于生成上下文。
- 展开后的 sources 只用于诊断或生成，不能作为检索指标的输入。

**经验**

我现在会先确认一个功能改变的是“检索结果”还是“结果展示形态”。父块展开属于生成侧内容替换，任何“展开后 MRR 提升”的结论都应先按测量口径错误处理。

## 上下文装填不能进入检索评估

**现象**

Context packing 会整块装填、跳过超预算块、去重并重排引用编号。如果这些动作进入 `retrieve()`，MRR 和 recall 就会跟着变化，检索 ranked list 也不再是原始排序结果。

**修复**

- `ContextPacker` 只在 `build_prompt` / `answer` 路径运行。
- `retrieve()` 返回的 sources 不做 context dedup 或预算裁剪。
- `RAGResponse.sources` 表示真正进入 prompt 的 used sources，不替代检索评估列表。

**经验**

上下文装填证明的是结构正确性：整块装填、预算复用、引用连号和 token 边界准确。它不应该产生检索指标收益，生成质量则需要单独的答案质量评估。

## 分隔符边界不等于句子边界

**现象**

早期超预算截断沿用了 TextSplitter 的全部 separators，其中包含逗号、分号和空格。测试只证明没有切半词，但实际文本可能停在 `...here. Gamma`，仍然属于句中截断。

**修复**

- 截断分隔符只保留段落与句终边界。
- 测试从“不切半词”升级为“必须停在完整句或段落末尾”。
- metadata 明确记录 `sentence_boundary`，不再使用宽泛的 separator 表述。

**经验**

我不能用“比旧实现更好”代替“满足真实约束”。词边界、子句边界和句子边界是三种不同语义，测试必须命中最终要求。

## RRF 融合首先要求候选 ID 对齐

**现象**

Dense 与 BM25 单路结果都正常，但融合结果没有预期收益。问题不在 RRF 参数，而在两路对同一 chunk 使用了不同 ID，导致融合器把同一内容当成两条候选。

**修复**

- Dense store 与 BM25Retriever 使用同一份 `VectorRecord` corpus。
- BM25 输出的 `RankedRecord.id` 必须等于 `VectorRecord.id`。
- 评估报告保留 metadata id，便于抽样核对各路径是否对齐。

**经验**

多路召回调权重之前，我会先核验候选 ID 空间。RRF 只认识 ID 和 rank，不理解两段文本是否其实来自同一个 chunk。

## 没有触发的 negative query 不能证明安全

**现象**

多查询离线 fixture 中，negative 样本与单路结果逐字节一致，表面上像是“没有退化”。实际原因是这些 negative query 没有任何改写，N=1 时整段 multi-query 被旁路了。

**修复**

- 文档明确标注 negative fixture 为零覆盖，而不是安全通过。
- 离线改写收益只描述为已知优质改写下的机制演示，不称为生产下界。
- negative 行为单独交给 abstention/fabrication 评估。

**经验**

功能没有运行，就不能拿结果不变证明它安全。尤其是 negative capability，我会检查功能是否真的被触发，而不是只看 overall 指标。

## `async def` 不等于真正异步

**现象**

FastAPI 端点已经写成 `async def`，并发请求却仍接近串行耗时。原因是函数体内仍直接执行同步检索、同步 chat 或 `time.sleep`，事件循环照样被阻塞。

**修复**

- 网络 I/O 改为 `httpx.AsyncClient`。
- retry backoff 使用 `await asyncio.sleep`。
- 既有同步检索漏斗通过 `asyncio.to_thread` offload。
- 用多个慢请求的总墙钟验证并发，而不是只检查函数签名。

**经验**

异步能力必须用时间行为证明。多个耗时 S 的请求并发后应接近 S，而不是 K×S；代码里出现 `async` 关键字本身没有证明力。

## SSE 不能先算完整答案再切片

**现象**

如果先调用 `answer()` 得到完整答案，再把字符串切成 token 逐段 yield，前端虽然能看到流式事件，但首 token 延迟仍等于完整答案耗时。

**修复**

- `/query/stream` 先发送 sources event。
- 后续直接消费 provider 的真实 streaming delta。
- 最后发送 done event，并验证 token 拼接结果等于完整答案。

**经验**

我判断流式是否真实，主要看首 token 时间，而不是响应类型。测试传输层可能缓冲数据，因此必要时要直驱 generator 或使用真实 uvicorn 探针。

## HTTP 异常映射不能依赖消息子串

**现象**

服务层曾通过 `str(error)` 判断空索引、维度不匹配和上游超时。当前文案能命中，但一旦有人调整异常文字，HTTP 状态码就会静默漂移。

**修复**

- 领域错误使用 typed exceptions。
- 上游 API/embedding 错误携带结构化 `error_kind`。
- HTTP 层只负责类型到状态码的映射。
- 错误响应统一脱敏，不暴露 key、header 或 traceback。

**经验**

异常映射是外部服务契约，不是日志字符串搜索。错误类型、可重试性和用户可见消息应该分开建模。

## Retry 循环不能反复创建 HTTP Client

**现象**

重试逻辑功能上可以成功，但每次 attempt 都重新创建 `httpx.AsyncClient`，连接池完全无法复用，额外增加握手、延迟和资源消耗。

**修复**

- 一次逻辑请求只创建一个 async client。
- 同一次请求的多次 attempt 复用该 client。
- 测试记录 client 创建次数，确认多次重试仍只创建一次。

**经验**

连接生命周期属于生产行为的一部分。功能结果相同，不代表资源管理正确。

## Redis corpus version 不能只存在进程内

**现象**

进程内 `cache_generation` 能让当前服务实例失效缓存，但 Redis 是跨进程存储。服务重启后 generation 归零，旧 L2/L3 key 可能重新变得可达。

**修复**

- `corpus_version` 存入 Redis。
- 入库成功后使用 `INCR` 原子递增版本。
- L2/L3 key 包含持久化 corpus version。
- `ServiceState.invalidate()` 同时清进程内 pipeline cache 和 bump Redis version。

**经验**

缓存版本必须和缓存本身拥有同样的生命周期。进程外缓存不能依赖进程内编号来保证新鲜度。

## Embedding 缓存不能使用 JSON Float

**现象**

JSON 序列化向量很方便，但浮点文本化可能引入细微差异。在相似度非常接近时，这种差异足以改变排序。

**修复**

- Redis 使用二进制 float64 序列化。
- payload 携带向量长度 header。
- 测试覆盖极值、非规格化数、`-0.0`、π 和高精度小数。

**经验**

向量缓存要求 hit 与 cold 路径逐 float 相同。“近似相等”不足以证明排序不会漂移。

## FakeRedis 不能证明真实 Redis 可用

**现象**

FakeRedis 可以证明 key、TTL 逻辑和 fail-open 分支，却不能证明依赖已安装、TCP 连接可用、真实 `INCR` 生效，或 `redis.asyncio` 的事件循环行为正确。

**修复**

- 默认 CI 保留快速的 FakeRedis 单测。
- 真实环境增加 gated smoke，覆盖 PING、SET/GET、TTL、INCR 和二进制向量往返。
- 再增加跨事件循环调用回归，复刻同步 CLI、离线评估和服务 threadpool 的真实形态。

**经验**

我会明确区分“替身测试”“单 loop live smoke”和“生产调用形态测试”。前两者绿灯不能替代最后一种证据。

## `redis.asyncio` Client 不能跨事件循环复用

**现象**

真实 Redis 开启后，L1/L2/L3 几乎不命中，日志出现 `Event loop is closed` 和 `Future attached to a different loop`。缓存 fail-open 后查询仍能返回，所以指标不漂反而掩盖了缓存已经失效。

**修复**

- Redis store 持有专属后台事件循环线程。
- async client 在该 loop 内懒创建并始终绑定同一 loop。
- 所有 Redis 操作通过线程安全调度进入 store loop。
- 回归测试要求第二次查询真实命中且 `errors==0`。

**经验**

异步 client 生命周期必须和事件循环生命周期一致。fail-open 只能证明服务没挂，不能证明缓存有效；命中率还需要上游调用计数和 Redis error 计数共同证明。

## 配置校验必须接入启动路径

**现象**

我曾经写好了 `Config.validate_cache()`，但启动入口没有调用它。非法 TTL 和缺失 URL 仍能带到运行时，这个 validator 实际上只是死代码。

**修复**

- 应用创建时执行对应配置校验。
- 测试覆盖非法配置启动即失败。
- 运行时外部服务故障与启动配置错误分别处理。

**经验**

配置校验只有接入启动路径才有价值。定义了函数不等于系统获得了保护。

## 请求体里的 Principal 不是可信身份

**现象**

即使 dense、BM25、多查询和缓存键都正确应用 ACL，如果服务直接信任请求体中的 `principal`，调用方仍然可以自报身份。

**修复**

- 默认只接受受信网关写入的 principal header。
- body principal fallback 默认关闭，只能显式用于本地测试。
- principal 作为显式参数沿 ACL 调用链传递，下游不再回读原始请求。
- 身份缺失或解析失败时 fail-closed。

**经验**

授权过滤不等于认证。生产部署必须让受信网关完成身份校验、剥离外部同名 header，再写入内部 principal。

## ACL 必须在候选评分前过滤

**现象**

先取 top-k 再过滤无权文档，会让高相似但无权的结果占满名额，有权结果被饿死；结果数量变化还可能泄漏无权文档存在性。

**修复**

- ACL 条件进入 dense similarity search 和 BM25 candidate scoring 前置过滤。
- memory、Qdrant 与 BM25 使用相同的集合相交语义。
- 有效 ACL filter 进入 L2/L3 cache key。
- 身份、解析或 store filter 能力异常全部 fail-closed。

**经验**

权限属于候选空间定义，不是结果展示过滤。任何检索路径或缓存路径漏掉 ACL 都可能成为旁路。

## MySQL ACL Schema 必须接进入库

**现象**

只创建 document、chunk 和 `acl_binding` 表，并不能证明 MySQL 已经成为权限真相源。如果入库仍只接受调用方手写 ACL，schema 只是半成品。

**修复**

- resolver 增加 source 到 ACL subject 的解析。
- 入库时可从 MySQL binding 解析 ACL 并写入向量 payload。
- 真实 MySQL 测试覆盖 binding 行到 chunk ACL metadata 的完整链路。

**经验**

MySQL 保存真相，Qdrant payload 只是执行快照。binding 变化必须通过 re-ingest 或 re-sync 进入 payload，不能宣称查询时实时强制。

## 观测故障不能变成查询故障

**现象**

如果 audit sink、metrics registry 或 alert hook 的异常冒回请求路径，正常查询会因为“记录失败”变成 500。ACL 的 fail-closed 语义不能机械复制到观测层。

**修复**

- audit、metric、alert emit 分别捕获异常。
- ACL 拒绝审计丢失时保留 WARN，但不改变原本的 403。
- JSON-line audit 使用有界后台队列。
- 测试注入会抛异常的 sink/registry/hook，确认查询结果不变。

**经验**

fail-open 还是 fail-closed 取决于组件是否属于安全边界。观测故障影响诊断能力，不应该改变业务结果；配置缺失则应在启动时早失败。

## 累计 Token 指标需要原子认领

**现象**

多个并发请求用 `after_total - before_total` 读取同一个 provider 累计计数时，会重复认领同一段 token。L3 缓存回放还可能携带旧 usage，再次把历史消耗记成新请求。

**修复**

- provider 暴露累计 request/token 计数。
- observability manager 按 provider identity 保存 watermark，并在锁内原子认领新增区间。
- request count 没增加时，不重复消费 cached raw usage。
- 上游有请求但无 usage 时才估算，并标记 `estimated`。

**经验**

`reported` 只说明数据来源，不保证聚合正确。累计计数必须有唯一认领点，缓存携带的历史元数据也不能当成本次新账。

## Ragas 与无上界 LangChain 版本不兼容

**现象**

依赖全部安装成功后，`import ragas` 仍然失败。`ragas==0.4.3` 还依赖 pre-1.0 LangChain 兼容模块，而解析器选择的新版本已经删除了这些模块。

**修复**

- 固定 `ragas==0.4.3`。
- 将 `langchain-community` 固定在 `<0.4`，其余 LangChain 组件固定在 `<1.0` 兼容范围。
- 增加四个 Ragas metric 的真实 import smoke，而不是只检查 `pip install`。

**经验**

“依赖安装成功”不等于依赖图可导入。对无上界的传递依赖，我会同时验证 resolver 结果和真实 import。

## 生成质量评估是一条数据出境通道

**现象**

Ragas live 评估需要 question、generated answer、retrieved context 和 ground truth。Answer Relevance 还会把文本发给 embedding endpoint，这些内容恰好也是审计和 ACL 系统重点保护的数据。

**修复**

- live 评估默认关闭，并要求第二个显式开关。
- fixture 只保存 hash、分数和 provenance，不保存原始评估文本。
- 每次 push 的 CI 只做离线 replay，不接触 provider key。
- ACL 敏感生产语料必须先完成外部处理方审批。

**经验**

评估工具不是天然安全的旁路。只要会发送上下文，就必须纳入数据出境治理。

## 托管 Embedding 模型会退役

**现象**

Gemini `text-embedding-004` 曾在 live 调用中返回 `404 NOT_FOUND`。继续查看模型列表后确认该模型已经退役，而 `supportedGenerationMethods` 也没有完整反映实际可调用方法。

**修复**

- 切换到 `gemini-embedding-001`。
- 真实验证 `batchEmbedContents` 可用，原生维度为 3072。
- 模型名、维度、`.env.example`、adapter、测试和文档同步更新。

**经验**

托管模型名是会过期的配置。接入或更换模型时，我会同时做 ListModels、真实 POST、维度与范数探针，不只依赖旧文档。

## 修改 `.env` 不等于运行时配置已改变

**现象**

磁盘 `.env` 已换成新模型，运行时 `Config` 却继续读取旧值。原因是 IDE 会话已经把旧环境变量注入父进程，而 `load_dotenv()` 默认 `override=False`，子进程继承的环境优先级更高。

**修复**

- 验证配置时读取 resolved `Config`，不只看文件内容。
- 需要即时生效时开启新终端，或在明确场景使用 `override=True`。
- 生产环境仍保持真实环境变量高于 `.env`，不全局反转 12-factor 优先级。

**经验**

配置文件、进程环境和解析结果是三层不同状态。最终核验只认运行进程实际解析出来的值。

## 结构化判官的 Token Budget 要按完整 JSON 计算

**现象**

Faithfulness 在 statement extraction 阶段抛出 `IncompleteOutputException`。最初的 4096 token 对普通样本足够，但完整 long-tail capture 在 q016 附近仍出现结构化输出截断。

**修复**

- 将 live capture 的 `RAGAS_JUDGE_MAX_TOKENS` 提升到 8192。
- 确认该参数被 Ragas 原样传给 provider，而不是配置未生效。
- 重新运行完整 capture，确保报告和 candidate fixture 能落盘。

**经验**

结构化输出预算要按完整 JSON 长度而不是可读答案长度估算。第三方库默认值只能当起点，长尾样本必须用真实运行验证。

## 生成温度和判官温度是两条随机链

**现象**

我曾经只把判官温度设为 0，却让答案生成继续使用产品默认 0.7。每次 capture 得到不同答案，fixture 固定的是一次随机抽样，而 metadata 中单一的 `temperature=0` 又容易误导为整条链确定。

**修复**

- `APIClient` 支持实例级 temperature override。
- live capture 单独使用 `RAGAS_LIVE_GENERATION_TEMPERATURE=0`，不修改产品默认 0.7。
- provenance 分别记录 `judge_temperature` 与 `generator_temperature`。
- 单测同时锁定显式 0 生效和未传值回退默认。

**经验**

可重放评估必须钉住每一个上游随机阶段。温度 0 仍不代表模型绝对确定，所以固定输入后的重复判官仍然必要。

## Mean 和 Stddev 不适合抵抗判官离群点

**现象**

Faithfulness 重复结果可能是 `[1,1,1,1,0]`。单个伪 0 会拖低 mean，stddev 又会把 margin 推高，导致一个离群裁决同时影响中心估计和噪声门禁。

**修复**

- live repetitions 使用奇数 5。
- case threshold 改用 median。
- margin 使用 MAD，mean/stddev/min/max 继续保留作诊断。
- aggregation method 与 spread method 写入 provenance。

**经验**

稳健聚合能吸收少数离群点，但救不了 `[0,0,1,0,0]` 这种多数派错误。统计修复只能处理噪声，不能替代判官协议修复。

## Faithfulness 伪零可能来自 Statement 过度原子化

**现象**

固定答案中的联合流程“call hotline and ask manager”曾被拆成两条 statement，每一条都隐含“单独动作就是完整 workaround”。上下文要求两项同时完成，因此 NLI 把拆坏的 statement 判 0 其实是正确行为。

**修复**

- 自定义 versioned statement-generation prompt。
- 保留 Einstein 示例，继续教模型拆分独立 claim。
- 新增联合流程示例，只在多个谓语共同构成一个要求、流程或定义时保留 conjunction。
- 真实 A/B 中 q012 从多数拆分变为 5/5 单条，Faithfulness 5/5 为 1。

**经验**

我不能从“答案忠实但分数为 0”直接推断 NLI 不可靠。必须 dump `生成 -> statement -> verdict/reason -> score` 每个中间态，先确认判官实际收到的 claim 是否仍保持原义。

## 引用声明需要 Source-Labeled Context 才能验证

**现象**

q004 的答案声明定义来自 `finance-policy.md [2]`，但 Faithfulness 只收到 chunk 正文，没有文件名和 citation 序号。statement 始终为一条，NLI 5/5 稳定判 0，因为来源归属在 content-only context 中确实无法验证。

**修复**

- Faithfulness 单独使用 `[n] filename: content` 格式。
- citation index 直接复用生成时的 `RetrievedSource.index`。
- 文件路径只取 basename，fixture 只保存 labeled-context hash。
- Context Precision/Recall 继续使用纯正文。
- 真实测试中 6 条 exact-name 样本 median 全为 1；构造的引错源反例仍被判低。

**经验**

这项修复把引用正确性纳入 Faithfulness，并不是剥掉 citation 放宽评分。正确来源要能被证明，错误来源也必须继续亮红灯。

## 门禁只应覆盖对生成结果敏感的指标

**现象**

Context Precision/Recall 的输入不包含 generated answer，它们量到的是固定 hash 检索 profile。把 0.70 阈值设为生成质量 gate，会让产品答案再好也被测试装置的检索地板卡住。

**修复**

- `GATED_METRICS` 只包含 Faithfulness 与 Answer Relevance。
- negative abstention/fabrication 继续独立 gate。
- Context Precision/Recall 降为 reported-only，仍完整输出 case、capability 和总体分布。
- gating scope 写入 provenance，漂移时 baseline invalidated。
- 检索输入变化通过 context hashes 确定性暴露。

**经验**

一个指标是否该 gate，取决于它是否消费被测变量。reported-only 不是藏数或放水，而是把门禁收回到真正能反映生成回归的信号上。

## Answer Relevance 的浮点尾差会越过 1

**现象**

完整 candidate capture 曾在 q005 中止。Gemini 向量 L2 范数约为 `1.0000000700`，近义问题点积约为 `1.000000139930`，Ragas 因此得到略高于 1 的 Answer Relevance。一个 epsilon 让整份报告和 fixture 都无法落盘。

**修复**

- capture 侧增加 `LIVE_SCORE_CLAMP_TOLERANCE=1e-6`。
- 只将容差内的有限值 clamp 回 `[0,1]`。
- NaN、inf、`1.5`、`-0.3` 等真实异常继续失败。
- replay 的 `validate_score` 保持严格，不接受任何越界 fixture。
- Gemini adapter 四个同步/异步入口增加 L2 归一化。

**经验**

数学合法区间和 IEEE-754 实际输出不一定逐位一致。容差应放在最接近不稳定来源的记录边界，并在持久化前收口；不能顺手放宽 fixture 完整性约束。mock 也必须覆盖 epsilon、非有限值和明显越界。

## 跑通评估装置不等于质量门禁通过

**现象**

clamp 修复后，Answer Relevance 80/80 repetitions 不再越界，但长尾 statement extraction 又暴露 4096 token 截断。将 budget 提升到 8192 后，完整 candidate capture 才第一次跑到底并得到真实分布：

- Faithfulness：mean 0.864，median 1.0。
- Answer Relevance：mean 0.847，median 0.887，MAD 0.089。
- Context Precision：mean 0.639，median 0.50，reported-only。
- Context Recall：mean 0.947，median 1.0，reported-only。
- Negative：abstention 0.5，fabrication 0.5。

报告能落盘，但生成敏感门禁仍有 9 条失败，因此不能冻结首份正式基线。

**修复**

- 没有为了得到绿灯而硬冻 candidate fixture。
- 将失败拆成三组继续诊断：Faithfulness 在固定答案下仍翻分、Answer Relevance 对部分 paraphrase/long-tail 稳定偏低、q018/q020 negative 需要区分真实虚构和拒答契约误判。
- 优先处理 negative，因为它属于安全关键行为。

**经验**

离线绿只证明装置自洽，live 跑通只证明装置能产出完整数据，两者都不等于质量过关。我会保留失败数据，先判断问题来自生成、判官、阈值还是确定性契约，再决定是否建立基线。

## 负样本弃答从 1.0 退回 0.5，是契约误判不是模型在虚构

**现象**

首份完整 capture 里负样本弃答率从 1.0 掉到 0.5：q018“宠物休假政策”、q020“季度工资税表”被判成虚构，q017 食堂菜单、q019 K8s autoscaler 仍正常弃答。“5/5 fabricated”一开始很吓人——看着像模型对库外问题稳定地编答案。

**修复**

我没有先动提示词，而是先把四条负样本在 temp0 下重新生成，连答案原文带检索 top-k 一起逐条导出来看。结果推翻了第一直觉：

- q020 其实是**有依据的拒答**——“The finance policy does not provide a payroll tax table… it does not list payroll tax tables [5]”，没有虚构内容，只是这句措辞落在六条弃答子串之外，被确定性契约误判成虚构。
- q018 这一轮直接干净弃答，说明上一轮的“虚构”是 temp0 生成漂移，不是真编了一条宠物政策。
- 检索确实把 employee-handbook、finance-policy 的相邻 chunk 喂了进来（近域），但这一轮并没有诱发真虚构。

定位成契约误判后，我选的修法是**把库外拒答钉成一句固定开头**，而不是去放宽子串：

- 系统提示词收紧成“只回答检索直接支持的内容、相关但不直接支持的不许外推”，并要求不支持时必须以固定一句 `The answer is not available in the knowledge base.` 开头。
- 弃答判定的首条 marker 直接由这句常量派生，让“生成措辞”和“判定契约”绑死、不再各自漂移。
- 没有放宽 marker：q020 的裸拒答、以及“借了 leave 就编出 5 天宠物假”这类近域虚构，仍然被判成虚构，并补反例测试守住口径。

**经验**

“5/5 fabricated”不代表模型在稳定虚构——负样本每条只生成一个固定答案，弃答判定是确定性的，5 次必然一致，真正会漂的是那一条 temp0 生成文本跨轮变化。一个只认固定子串的弃答契约，只有在生成被钉到同一套措辞时才是可靠测量；与其迁就坏样本去放宽口径，不如把“库外→固定拒答句→命中契约”这条链本身做稳。真打验证我连跑两轮、8 条全新 temp0 生成全部弃答，q020 现在以固定句开头并保留引用——但 temp0 仍非确定，这是稳定性证据而非确定性证明，残留我留在记录里。这一摞解决后，冻结正式基线仍卡在 Faithfulness 与 Answer Relevance 两摞。

## Faithfulness 固定答案翻分，根因在 statement 抽取不在 NLI 裁决

**现象**

首份完整分布里还有一摞没解：Faithfulness 在固定答案上翻分。8192 capture（每条 5 次重复、判官 temp0）里，q008、q012、q013、q016 四条正样本的 Faithfulness 在 5 次重复间剧烈摆动，其余 12 条稳定 1.0。关键前提是：正样本的答案在一次 capture 内只生成一次，5 次重复是对同一条固定答案、同一组 context 反复判官。答案和 context 都没变、分数却翻，变量就只能落在判官侧——要么 statement 抽取不稳，要么 NLI 裁决不稳。

**修复**

我没有先改提示词，而是先把判官链逐 rep dump 出来看。dump 显示只有 q016 复现了翻分；q008/q012/q013 在重新生成的答案上没有再翻，而 fixture 只存 hash 不存原文、历史输入无法逐字节重构，所以这三条我只能诚实记成“对当前冻结输入稳定、历史翻分无法复现”，不强行宣称已修。

q016 复现后，逐 rep 对比把根因钉在抽取层：答案是一句 “Source [4] is relevant because <具体程序>”，抽取时会间歇地把它原子化成一条裸的 “Source [4] is relevant.” 再加上拆出来的程序句。而 Ragas 的 NLI 只收到 context 和 statement、看不到原始问题，那条裸“相关”根本无法判断“相关于什么”，于是被稳定判 0；理由不被切开时，同一条关系反而被稳定判 1。这不是答案不忠实，是抽取把可判证据剥掉了。

我先尝试只强化 prompt，但 v2 对同一条旧答案仍得到 `[1,1,1,.5,1]`：失败轮虽然保留了 `relevant because`，却把真正证明相关性的具体 procedure 拆到了另一条 statement，剩下的理由仍然不自足。这证明 prompt-only 还没有闭合。

最终修法升级为 `question-independent-rationale-preservation-v3`，并加一道很窄的确定性后处理：只对“单句且含 `is relevant because`”的答案，把整句还原成一条自足 statement。NLI 提示和裁决一字未动，并补了一条反例测试，确保“完整但未被 context 支持的相关性声明”仍然判 0，不是靠保留整句来放分。

**经验**

我用同一条冻结的 q016 答案和同一组确定性 contexts 做了逐版对照：v1 是 `[.667,1,1,.667,.667]`，prompt-only v2 是 `[1,1,1,.5,1]`，v3 则连续两轮共 10 次全部为 1，而且每次都把完整理由保留在同一条 statement 中。q008、q012、q013 的冻结答案在 v3 下也各自 5/5 为 1；其中 q013 偶发把 citation `[3]` 单独抽成 statement，虽然没有影响本轮分数，我仍把它记作残留诊断噪声。

这次对照没有证明 DeepSeek temp0 判官变成了确定系统，只证明窄 guard 消除了已知的 statement 边界随机变量。最大的教训是：prompt 修复也必须用同一条旧输入回放验证；换一条更短、更好判的新答案得到满分，不能证明原故障被修好。判官多步管线排障必须逐层 dump 中间态，也必须保留“prompt-only 失败”的中间版本，避免把相关性误当因果。这一摞解决后，冻结正式基线仍欠 Answer Relevance 长尾真低和一次全量重新 capture。

## 把托管 Embedding 换成本地模型，先问清它到底服务谁

**现象**

answer_relevance 这一路原本走托管的远程 embedding。某天连续两次全量 capture 都被日配额 429 挡死：20 条答案全生成、0 条 embedding 成功、十次退避也救不回，冻结正式基线卡在配额、不在质量。作为一个用来展示的项目，我不愿意让一条离线评估闸长期依赖一个会因为额度、网络或服务退役而随时打不通的远程 embedding。

**修复**

我把 answer_relevance 的 embedding 换成本地 fastembed（ONNX runtime、不引 torch），provider 仍实现原来的四个方法、L2 归一化沿用原 adapter，Ragas 装配一行不动。provider 走 `{gemini, bge}` 白名单仍 fail-closed，本地模式不再要求远程 key/url；换 embedding 是基线失效的协议变更，靠 provenance 里的 provider/model/dimension 自动拦下旧基线。

中途我为了“中英文兼容”先选了一个双语模型，真打 capture 直接否决了它：answer_relevance 在英文评测集上挂了 4/16，其中一条同义改写问句（“Which checks should support run” 对上 “What should support check according to the runbook”）被判成近乎无关、分数 0.37，低于跑题地板，连调阈值都救不回。我把判官生成的 reverse-question 逐条 dump 出来，确认它们语义都对——是模型本身在低词面重叠的英文改写上塌缩；换回英文模型重嵌同一批 reverse-question，这两条从 0.16 / 0.37 回到 0.91 / 0.83。

**经验**

最大的教训有两条。

其一，离线的语义基线是必要不充分的。我先用“等价改写 vs 跑题改写”量过一遍标度，看起来 0.70 阈值天然分得开——但那批改写词面重叠高，恰好掩盖了真正的失败模式。真闸里判官生成的 reverse-question 改写得更激进，只有喂进端到端真打 capture 才暴露塌缩。语义基线能排除一些可能，却代替不了真跑。

其二，换组件前要先问清“它到底服务谁”。这条 answer_relevance embedding 只用来给离线英文评测集打分，产品检索根本不消费它。所以把它换成双语模型，既不会让产品具备中文检索能力（那是另一条产品侧的检索 embedding），又实测拖低了英文评测——双语在这里零增益、反而伤准确度。需求是真的，但放错了层：中文要落地，该动的是产品侧的检索 embedding，而不是这条评估闸。顺带一个收益：本地推理是确定性的，同一条答案多次重复判分的离散度比远程 embedding 明显收紧，这对依赖 within-case 离散度设计的门禁 margin 反而更友好。

## 改生成提示词调 Answer Relevance，每类问题都有一个“形状中间量”

**现象**

Answer Relevance 的判分是判官从答案反推几条问题、再和原问算 cosine 取均值。我一开始以为“答案写得越完整越好”，真打下来发现它对答案的形状和信息量非常敏感，而且不是单调的：exact-name lookup 这类只回一个裸名字加引用，反推出来的问题信息不足，cosine 会塌，甚至被判 noncommittal 直接清零；反过来给所有答案都套上关系脚手架或场景条件，窄问题的反推又会漂离原问，cosine 同样掉。“更详细”或“更简洁”都不是全局答案。

我犯过的更大的错，是用一处全局生成提示词改动去救某一条长尾低分（比如 workaround 那条）。离线门禁绿、抽几条小样本也绿，但全量真打一跑就发现它把原本健康的样本打穿了——一处提示改动会同时牵动多个问题、多个维度。更隐蔽的是两个 gated 维度会在同一条答案上对冲：给场景类问题补条件能抬高 Answer Relevance 的 scope，却让模型自补的条件失去检索上下文支撑，直接打穿 Faithfulness。

**修复**

我停掉了“全局调松紧”的思路，改成按问题形状分桶路由生成提示，每一桶都校准到“答案 + 最小支撑短语”这个中间量：

- 名称 / 来源 / 关系实体类查询：保留一句“X 是 Name [n]”的最小关系骨架，既不留裸名、也不堆邻近职责。
- workaround 类：保留完整动作加触发条件，这类要的是更多内容而不是更少。
- 场景检查类：用上下文原句的动词、不让模型自补条件，避开与 Faithfulness 的对冲。

我连续改了几版生成提示，每版都做全量真打加逐桶对比，失败集每轮位移、收敛，直到一份 capture 里 16 条正样本的 Answer Relevance median 全部过 0.70、Faithfulness median 全部为 1.0、负样本全部弃答。然后我没有只凭一份 capture 就收口，而是又独立跑了第二份确认：第一份里贴着 0.70 的两条、以及一条 Faithfulness 单次闪到 0.5 的样本，在第二份都收住了。最后把这份通过的 capture 提升成正式冻结 verdict fixture（pin 生成提示版本、statement 提示版本、judge model、embedding 与 schema），离线 replay 自证门禁通过。

**经验**

Answer Relevance 不存在“答案越详细越好”或“越聚焦越好”的全局方向，每一类问题都有一个不偏不倚的形状中间量，要按问题类别去校准而不是动全局松紧。一处生成提示改动牵动多 case 多维度，离线绿和小样本绿都不算数，必须全量真打复核；两个 gated 维度还会在同一条答案上互相对冲，调一边要盯另一边别塌。judge 温度 0 仍然非确定，单次闪低用 median 门禁加 MAD margin 吸收，并且连跑两份独立 capture 看稳，而不是抽中一次就冻基线。用产品提示去拔评估分数，我守的红线是不下调阈值、不为迎合反推问题去改答案；目标始终是答案对问题本身合适，分数只是副产品。

## 一键起全栈的第一关，是镜像能构建而不是代码能跑

**现象**

我把服务打包成镜像时，第一次构建卡在依赖安装阶段，回溯了一个多小时才失败，报的是依赖解析空间过深。本地虚拟环境里同一套代码跑得好好的——问题不在代码，在于打包用的依赖清单全是无上界的 `>=`，再加上一个产品根本没 import、只是历史遗留的重依赖，把版本求解空间炸开了。

**修复**

我删掉那个用不到的重依赖（它对应的功能其实是走 REST 直连、根本不引 SDK，删了零功能损失），并把其余依赖钉到本地已验证通过的确切版本，重新构建就秒过了。我还意识到，交付一个镜像前如果一次干净构建都没跑过，这种解析爆炸永远不会在开发机上出现——因为开发机的解释器早缓存好了一套可用版本。

**经验**

“代码能在本地跑”和“镜像能构建出来”是两件事，一键起全栈的第一道门是后者。给服务单独维护打包依赖清单时，要钉死版本、删掉不 import 的依赖，并在交付前真跑一次干净构建，别把版本求解留给部署那一刻。

## 以脚本路径跑初始化，项目根不在导入搜索路径上

**现象**

容器启动脚本用“脚本文件路径”的方式跑一次性初始化，结果初始化一上来就报找不到项目里的顶层包。包明明在镜像里、文件也确实拷进去了。

**修复**

根因是解释器把“脚本所在目录”放在导入搜索路径的第一位，于是子目录在路径上、项目根反而不在，所有顶层包都导不进来。改成显式声明项目根为导入根（或用模块方式运行入口），初始化就能导入全部包。又因为初始化在启动即退非零、后面的服务进程根本不会被拉起，这个坑会直接表现为“容器起不来”。

**经验**

“文件在镜像里”不等于“能被导入”。以脚本文件方式跑子目录里的入口时，导入根会跟着脚本走、而不是项目根；一次性初始化、迁移这类脚本尤其容易中招。要么显式固定导入根，要么用模块方式运行。

## 自包含镜像不能依赖运行期临时去外网拉资源

**现象**

镜像在我机器上跑得好好的，换个网络环境首启就崩：某个分词器在第一次使用时会去公网下载一份编码表，下载被网络波动打断就直接抛异常、进程退出。因为它只在“首次使用”触发，构建和冒烟都可能碰不到，纯粹是“在我机器上能跑”。

**修复**

我把这份编码表在构建阶段就预热进镜像的缓存目录、并让运行用户可读，运行期零外网。验证方式是最较真的一步：用完全断网的方式起容器跑一次分词，确认零外网也能成功。代码里那条“下载失败就降级”的兜底虽然也值得加，但部署镜像的正解是构建期把资产打进去、让组件按设计工作，而不是静默降级。

**经验**

“自包含镜像”要求所有运行期资源在构建期就位。任何“首次使用时去网上拉一份”的隐藏下载，都是隔离网/受限网部署的定时炸弹，而且极易被“开发机有外网”掩盖。把这类资产在构建期固化，并用断网起容器这种最狠的方式验证，才算数。

## 就绪探针必须“失败即关闭”，和观测的“失败即放行”正好相反

**现象**

给服务加健康检查时，最容易犯的错是把“健康”写成一个恒返回 200 的浅探针，或者在探测后端时用 try/except 把异常吞掉、继续返回 200。这样一来后端明明挂了，流量还在往这个实例打。

**修复**

我把健康检查拆成两个语义不同的探针。一个存活探针只表示进程还活着、恒返回 200，进程没死就不该因为后端抖动被重启。一个就绪探针真去探当前配置下启用的后端（向量库、缓存、权限库），任何一个必需后端不通就返回 503 并点名是谁挂了。探测要并发、每个探针有独立超时上界（避免一个后端卡死把整个就绪检查拖挂）、并且按配置条件探测（没启用的不探、启用的必探）。探向量库时我只做只读连通查询，绝不走会建集合的构造路径，免得探活反而制造副作用；探缓存时走缓存组件自己绑定的那条稳定事件循环，不从请求线程裸调异步客户端——这正是缓存那一层早先踩过的跨事件循环坑。

判定这套东西是否真的失败即关闭，唯一算数的证据是真把后端一个个停掉：就绪探针要真的翻 503 且点名、存活探针要仍旧 200、后端拉回来要能自动恢复 200。用 mock 造一个“假装挂了”的后端不算数。另外容器编排里的健康检查我特意仍然打存活探针而不是就绪探针——否则某个后端短暂抖动会让编排以为进程坏了、把好好的进程重启掉。

**经验**

存活和就绪是两个方向相反的语义：存活是“别没事找事重启我”、偏宽容，就绪是“后端不行就别给我导流量”、必须严格失败即关闭。这和可观测性那一层的“失败即放行”正好是镜像关系，写的时候要时刻分清自己站在哪一边。就绪探针还有三条底线：要有超时上界不能挂起、响应体里不能带连接串/密码/密钥这类敏感信息、“探到后端连通”不能用“配置校验通过”来冒充。而这一切只有真断后端才测得出来。

## 把配置校验接进应用启动，导入这个模块就会触发校验

**现象**

我把几处结构性配置校验从“第一次用到才报错”提前接进了应用工厂，让非法配置在启动时就快速失败、而不是拖到第一个请求。改完发现一批原本能直接导入的测试开始在导入阶段就报错。

**修复**

根因是模块顶层有一行“导入即创建应用实例”，而应用工厂现在会跑这些校验；于是“导入这个服务模块”这个动作本身就带上了校验副作用，缺配置的测试环境一导入就炸。修法是让这些测试在导入服务模块之前先把所需配置置成可用值（比如把 embedding 切到不需要密钥的测试实现）。同时我特意没有把“需要付费密钥”的那条生成校验也接进启动——否则没有密钥连服务都起不来，会打穿此前特意保住的“无密钥也能起栈跑入库和检索”的冒烟路径；缺密钥应表现为就绪探针里那条生成依赖降级，而不是启动失败。

**经验**

把校验前移到启动是对的（非法配置早失败好过运行时才炸），但要意识到它的连带效应：只要模块顶层有“导入即建实例”，校验就会变成导入副作用，任何导入这个模块的地方都得先备好配置。前移时还要分清哪些是“结构性、缺了必错”该拦在启动，哪些是“缺了只是某功能不可用”该降级为就绪状态——把后者也拦在启动，会连带打穿无密钥冒烟这类刻意保留的能力。

## 扫描版 PDF 不能静默入库成“空知识”

**现象**

扫描版 PDF 没有文本层，普通 PDF 文本抽取只会拿到空字符串。更糟的是，我的 PDF loader 会给每页加 `--- Page N ---` 页眉，所以纯扫描 PDF 看起来“加载成功了”，实际入库内容只有页眉骨架，没有任何业务知识。这类错误不会在 embedding 或检索阶段自然爆出来，因为下游看到的是合法字符串。

表格抽取还有另一个隐蔽问题：每处理一页就重新打开一次 pdfplumber。小 PDF 看不出问题，一旦页数变多，就会把整份文件解析重复做成按页线性放大的开销。更危险的是，表格开关虽然存在于 loader 构造参数里，却没有从 CLI 和服务入库入口真正接线，修 loader 本身等于修死代码。

**修复**

我把扫描页检测变成默认行为：只要某页文本层低于阈值，就计入 `scanned_page_count` 并打 warning；OCR 仍然默认关闭，只有显式开启时才对这些空文本层页面逐页兜底。混排 PDF 不能整份二选一，必须只 OCR 真正没有文本层的页，避免把已有文本层的页重新识别出不稳定内容。OCR 依赖也做成可选 gated：没开 OCR 时不导入；显式开了 OCR、遇到扫描页但缺依赖，就直接报错，而不是假装成功。

表格抽取改成在一次 `load()` 内惰性打开 pdfplumber，一份 PDF 只打开一次，再按页复用。为了证明不是靠秒表碰运气，我用 spy 直接数 `pdfplumber.open` 调用次数；为了证明默认路径没漂，我用多页 PDF 的表格开/关输出做字符级回归。CLI 和服务入库入口都走同一个配置构造函数，PDF 表格/OCR 开关从环境配置进入 loader，不再停留在死参数里。

**经验**

入库层的第一责任是诚实：不能把“没有抽到内容”伪装成“成功加载”。扫描页、乱码、二进制文本和表格开关这类问题都发生在检索之前，等到检索指标再发现已经太晚。对这类能力，默认路径要保持可复现，增强路径要显式开关，失败契约要分清楚：可选表格依赖缺失可以 warning-skip，显式 OCR 依赖缺失却必须报错，因为用户已经要求用 OCR 兜底。性能优化也要用结构证据证明，像 `open()` 调用计数这种证据比计时更可靠。

## 冻结基线的 hash 绑定被行尾差异打了假阳性

**现象**

Ragas replay 门禁用 golden set 文件的 sha256 作为 `golden_version` 绑定，防止 fixture 与 golden set 脱节后还在“复现”一份过期基线。这个绑定一直安静，直到某天在 Windows 检出上跑标准 replay 突然报 `golden_version does not match`——但我逐条对比 golden set 内容，与录制 fixture 时一模一样，没有任何真实改动。

排查分三步。先在干净 HEAD 的独立 worktree 里复现，排除工作区未提交改动的嫌疑；再把 fixture 里记录的 hash 与 git 对象库里 LF 版本 blob 的 sha256 对比，两者逐位相等；最后把工作树文件行尾归一化成 LF 再喂给 replay，门禁立即通过。结论清楚了：git 的 autocrlf 在 checkout 时把 LF smudge 成 CRLF，`git status` 显示 clean（git 按归一化内容比较），但文件原始字节已经不同，对原始字节做的 sha256 自然漂移。同一份“逻辑内容”，在不同 OS 检出后字节不相等。

**修复**

主修是让 hash 只对逻辑内容敏感：hash 前把 `\r\n` 归一化成 `\n`，其余字节一位不动。刻意不做更宽的归一化——strip 空白、JSON 规范化、行排序统统不做，因为归一化每宽一分，漂移检测就钝一分，真实改动被抹平的风险就大一分。录制侧和校验侧必须共用同一个归一化函数，否则两侧口径分叉，未来录制的 hash 换个 checkout 又会炸一次。

`.gitattributes` 给 golden set 钉 `eol=lf` 只能作叠加防御，不能当主修：即使钉了，编辑器把工作树文件写回 CRLF 时 `git status` 依旧 clean，字节 hash 照样漂。修文件属性治标，修 hash 口径才治本。

流式实现还有一个容易漏测的坑：归一化按块流式做时，`\r\n` 可能恰好被块边界劈开——前一块结尾是 `\r`、后一块开头是 `\n`——需要把悬挂的 `\r` 递延到下一块再判断。单测若只用小文件，这条路径永远不会被踩到。我专门构造了跨块边界的 `\r\n`、块尾 `\r\r\n`、文件结尾孤 `\r` 等边界样例，与“整读后 replace 再 hash”的朴素参照逐位对拍，确认流式版与参照等价。

**经验**

用原始字节 hash 做”内容没变”的绑定，等于把门禁耦合到了 checkout 的行尾配置上——绑定的锚点应该是逻辑内容，不是存储字节。修这类假阳性最重要的是保住 fail-closed：修完必须补一个反例（内容真改一个字符，门禁仍拒绝、非零退出），否则很容易”修好了假阳性、顺手修死了真告警”。还有一条硬约束是冻结基线文件本身一个字节都不动：hash 口径修对了，旧 fixture 里记录的 hash 应该原地转绿；如果修完还得重录基线才能过，说明改的是错的地方。

## 相关性门不能跨分数空间共用一个阈值

**现象**

我想在检索层提前拒绝域外问题，第一直觉是给 `source.score` 设一个阈值。但 `source.score` 随配置变的是不同的物理量：纯 dense 是 cosine，开 rerank 是 reranker 分，开 hybrid 是 RRF 融合分。我在 20 条 golden 上把五个分数空间的正负样本区间都量了一遍，没有一个能被单阈值干净分开。

最致命的是 RRF 融合分：它只读 rank、从不读底层分数，正负样本区间四位小数完全重合（都是 0.0164–0.0197）。更隐蔽的陷阱是跨空间共用常量——RRF 最大值 0.0197 比 BM25 最小值还小两个数量级，比 lexical 最大值小三个数量级。任何一个数字放进去，要么在 RRF/cosine 空间全拒，要么在 BM25/lexical 空间全不拒。

**修复**

- 一个分数空间一套配置键，禁止 `RELEVANCE_GATE_MIN_SCORE` 这种跨空间键。
- RRF 融合分永久不作为判据；hybrid 开启时闸骑在被保留的 `metadata[“dense_score”]` 上，永不读 `source.score`。
- 不可判 ⇒ 不拒 + 出声：BM25、lexical、hash、未知 reranker 这些空间闸必须 no-op 并留下计数/日志，禁止静默跳过。
- 默认关闭、不设产品默认阈值：最窄的 bge 窗口只有 0.0263 宽，n=4 撑不起一个常量，阈值作为部署参数在 eval profile 里显式开。
- 报告必须同时给”拒绝收益”和”反向代价”：只报 negative_empty_rate 而不报 positive_false_abstention_rate 的单向指标必然”成功”，我不采信。

**经验**

给一个随配置变的物理量设单一阈值，等于把门禁耦合到了一个会换尺子的量上。跨数量级的分数空间绝不能共用常量。机制做实不等于可以宣称质量——n 个样本撑不起一个产品默认值，reported-only 的 fixture 数字不能外推成生产阈值。还有一个反复出现的教训：不可判时必须出声，静默 no-op 会让”功能开着其实从不触发”的假绿活下来，和负样本零覆盖那个坑同源。

## 我目前仍在重点关注的边界

**现象**

系统已经具备较完整的检索、缓存、权限、服务和评估能力，但仍有几类风险没有自然消失：negative query 行为、ACL 权限同步窗口、全局 Redis corpus version 的过度失效、多 worker 指标聚合，以及流式缓存回放的 raw response 溯源差异。

**修复**

- negative 弃答现在分两层：生成层靠固定拒答句 + 契约（残留是 temp0 可能漂掉固定句），检索层新增默认关闭的相关性门（机制做实、阈值只在 eval-only bge profile reported-only 跑过，产品阈值仍需在更大数据集上校准）。两层不互相替代。
- ACL binding 变化需要明确 re-ingest/re-sync 策略。
- Redis corpus version 后续可按 collection 细化。
- 多进程指标交给外部 Prometheus 聚合。
- 缓存流式 replay 明确标注，不冒充 live generation。

**经验**

面向企业的系统没有“功能完成就自然生产可用”这一步。每个能力都需要明确生命周期、失败语义、观测边界和真实环境证据，我会继续按这个标准推进。
