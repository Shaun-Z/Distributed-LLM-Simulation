# 实现与验收状态

对照 `heterogeneous-distributed-simulator-design.md` v1.0，当前交付为可运行研究原型。原始设计保留，不把计划中的所有可选研究扩展改写成完成状态。

| 阶段 | 已交付 | 验收依据与边界 |
|---|---|---|
| P0 | 不可变 schema、单位、规范化/哈希、discrete/UMA/two-host profile、三类非 LLM builder | 非法输入拒绝、配置 round-trip、图导出；数值全为 synthetic |
| P1 | 稳定 heap、startup/body、同刻完成优先、零时长 closure、固定 compute/copy、slot、可重放 trace | 单请求 6.1 ms；两请求 9.1 ms；独立/共享槽 fork 2/4 s；长期时间与高带宽舍入回归 |
| P2 | 物理 pool 归一、对齐整数分配、原子多 pool 输出/workspace、pin/refcount、复制去重、staging、输出回传 | UMA 6 GiB/8 GiB、fan-out 一个副本、OOM 无部分提交、source 到最后读取完成才释放 |
| P3 | transfer weighted max-min、逐 channel limiter、共享物理预算、generation、服务/容量积分、资源顺序解释 | 两流在 7/11 s 完成；parking-lot 5/5/95；full/half duplex；ring 理想 10.2 s |
| P4 | 分段常量轨迹、暂停恢复、独立 RNG、预生成 Markov 网络/Poisson arrival、独立 run 统计、测量格式、copy 拟合与留出验证 | 100→50 带宽 oracle 为 15 s；暂停恢复为 13 s；重放查询不采样；真实设备验证仍待测量 |
| P5 | fork-join/image/tiled GEMM；Transformer prefill/decode、GQA、append KV、TP collective、PP microbatch、按需 offload 重载、LLM marker 指标 | 相同 engine 无 LLM 分支；TP/PP/多请求测试；offload 增加流量并降低设备常驻权重峰值；动态 serving 扩展另列 |
| P6 | evaluator/cache、placement 枚举、rank 贪心规划、budget sweep、经验 Pareto、条件/全局下界、gap、有限 A 层调度 oracle | 31 s FIFO 的小图可证明固定计划最优为 22 s；预算耗尽只报告 incumbent/bound；不证明全局 route/placement/变换最优 |

## 模块路径

- `spec/`：静态语义、输入格式与能力校验。
- `compiler/`：通用执行图、合法路由/多段 staging、collective 展开。
- `models/`：计算成本、原子内存管理、外生网络过程、校准接口。
- `runtime/`：事件、活动状态、仲裁、不变量与仿真时钟。
- `scheduling/`：FIFO、critical rank、memory-aware、oracle start order。
- `workloads/`：所有领域特定内容，仅生成通用对象。
- `analysis/` 与 `io/`：指标、解释、bounds/search、完整实验产物。

ready 候选由 predecessor 计数与 successor 索引驱动。分配释放、完成、到达与容量事件触发候选检查，不使用墙钟轮询。当前在事件边界重检 ready 候选并重算活跃流，保留显式 blocker 现场；尚未做全量 resource→waiter 增量调度优化。trace 目前驻留内存，适用于研究原型规模；大 trace 可通过级别裁剪。

## 明确的模型边界

1. Compute 是不可抢占、固定时长，并独占所需内存服务组；共享算法仅处理 byte-rate transfer。时变 kernel 内部 IO 需要未来的 phased/normalized-progress 模型，当前拒绝混用。
2. Startup/传播为固定 phase，保守持有 slot/exclusive gate；传输端到端流体服务，不模拟包、TCP、store-and-forward 协议或无线 MAC。多个 memory endpoint 的 staging 是多个独立 whole-tensor hop。
3. Replica 只表达不可变版本，禁止同活动输入/输出 allocation 重叠。当前没有一般 view/alias byte range；UMA 地址视图共享物理 pool，不能当成另一份容量。
4. 显式 reload/offload 采用每次消费者读取后释放临时副本；不是自动 eviction、重计算或双缓冲最优策略。没有承诺任意 scheduler 的时间随内存预算单调。
5. Transformer 是普通/GQA、合成成本、未融合 decoder 模型。KV 是初次 prefill 后逐 token 增量 block，保留至 run 结束。没有 embedding/vocabulary sampling 的精确计算、动态 batch fusion、分页/共享前缀/COW 或真实模型权重加载。
6. Collective 支持显式 ring 消息和 reduction/local pack IO；relay sync 是收到 whole chunk 后的元数据交接。没有 NIC 内核、in-place alias 或网络内 reduction；缓冲区保守分开分配。理论 ring 测试显式关闭局部计算/pack 成本，并固定合法直达路由。
7. Oracle 只证明固定 A 层 execution 的 schedule 最优；枚举全体拓扑 start order，按顺序最早启动，允许等待未 ready 的 head 来主动空闲。常量、无抢占、独占服务保证给定 start order 的最早启动不劣于延迟启动；这个论证不适用于 fluid sharing/时变资源/动态 eviction。达到枚举或仿真上限时不报告证明完成。
8. 全局 compute 依赖下界只在固定 workload/算法变换集合与乐观 profile 前提下成立。执行图负载界固定 placement/routes。没有泛化 Hong–Kung 定理、通用异构 cut LP 或完整多目标最优 frontier。
9. 未实现故障/重试、checkpoint、能耗/成本模型、抢占、动态 DAG_APPEND/continuous batching。有限请求与自回归序列预先展开；公开策略只见当前只读 snapshot，不能读取未来环境。

## 可复现验收

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
uv run heterosim demo pipeline --output runs/acceptance
uv run heterosim replay runs/acceptance
uv run python examples/research.py --output runs/research-acceptance
```

结果目录遵循第 21.4 节，额外保存 execution、timeline、graph、critical_path 和静态 bounds。`failure_snapshot.json` 只在无进展/截断时存在。`oracle/best` 保存所选 start order，可用普通 `replay` 复现；scope 和穷尽标志写入 `oracle.json`。

真实校准必须提供真实硬件与软件标识、测量时间/方法/来源、同步完成时刻，并分离 calibration/validation。当前测试验证合成数据拟合及留出误差流程，没有据此宣称经验精度。

## 可视化工作台

已新增英文 React/TypeScript 工作台与可选 FastAPI 服务，入口为 `uv run --extra web heterosim ui --runs-dir runs`。支持表单与高级配置、校验预览、单次仿真队列与取消、拓扑/DAG、时间线回放、资源曲线、trace/失败诊断及最多四个 run 对比。CLI 与 Web worker 共用运行和导出应用服务，不改变引擎语义。

结果图表读取已有产物；不提供伪实时进度或从缺失 trace 推测曲线。既有 CLI、sweep/oracle 子目录可只读浏览。首版不包含拖拽建图、搜索启动界面、账户和协作功能。运行与打包见 [工作台指南](workbench.md)。
