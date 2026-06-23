# 核心踩坑记录：我的企业级 RAG 工程复盘

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

## 我目前仍在重点关注的边界

**现象**

系统已经具备较完整的检索、缓存、权限、服务和评估能力，但仍有几类风险没有自然消失：negative query 行为、ACL 权限同步窗口、全局 Redis corpus version 的过度失效、多 worker 指标聚合，以及流式缓存回放的 raw response 溯源差异。

**修复**

- negative 行为继续做答案原文诊断和 abstention 契约核对。
- ACL binding 变化需要明确 re-ingest/re-sync 策略。
- Redis corpus version 后续可按 collection 细化。
- 多进程指标交给外部 Prometheus 聚合。
- 缓存流式 replay 明确标注，不冒充 live generation。

**经验**

企业级系统没有“功能完成就自然生产可用”这一步。每个能力都需要明确生命周期、失败语义、观测边界和真实环境证据，我会继续按这个标准推进。
