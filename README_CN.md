# Distributed LLM Simulation

[English](README.md) | 简体中文

按 [工程规格](docs/heterogeneous-distributed-simulator-design.md) 实现的异构分布式计算研究原型。核心通过 **硬件资源图、Workload DAG、Placement、Scheduler 和离散事件引擎** 模拟通用计算；LLM 由独立插件展开。Python 3.12+，依赖与开发工具统一用 `uv` 管理，确切版本记录在 `uv.lock`。

## 安装和演示

在仓库根目录执行：

```sh
uv sync --locked
uv run heterosim demo pipeline --requests 2 --output runs/pipeline
uv run heterosim inspect runs/pipeline --timeline
uv run heterosim replay runs/pipeline
```

Pipeline 的单请求结果为 **6.1 ms**，两请求结果为 **9.1 ms**，与设计文档第 14 节一致。重复运行请选择新目录，或显式加 `--overwrite`。

```sh
uv run heterosim demo fork_join --output runs/uma
uv run heterosim demo image --output runs/image
uv run heterosim demo matmul --output runs/matmul
uv run heterosim demo transformer --output runs/transformer
uv run heterosim demo transformer --offload-pool ram0 --output runs/offload
uv run python examples/research.py --output runs/tp-pp
```

`distributed-llm-simulation` 和 `python -m distributed_llm_simulation` 也可作为入口。`demo` 使用仓库的 `profiles/`；安装 wheel 后可通过 `run` 传入自己的配置。

## 可视化实验工作台

英文 Web 工作台支持表单与 YAML/JSON 配置、独立进程运行及取消、硬件拓扑与逻辑/执行 DAG、时间线回放、物理内存和服务速率曲线、trace/失败诊断，以及最多四个实验的对比。

```sh
uv sync --extra web --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --extra web heterosim ui --runs-dir runs --port 8765
```

右上角 **Light theme / Dark theme** 按钮可切换明暗主题，选择会保存在本地，刷新后恢复；拓扑/DAG、图表和源码编辑器同步切换。暗色 DAG 的连线、箭头、标签和选中轮廓已提高对比度。

打开 **http://127.0.0.1:8765** ，选择真实内置场景，编辑、校验并启动实验。已有 CLI 结果只读加载；任务状态实时更新，仿真图表在运行结束后读取完整产物。缺失事件时明确显示数据不可用。Node 只用于开发和构建；发布 wheel 包含前端静态资源与演示 profiles。

详细操作、TP/PP 示例、曲线语义、开发测试与打包方式见 [工作台指南（英文）](docs/workbench.md)。

## 配置和 CLI

```sh
uv run heterosim validate --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --export runs/validated
uv run heterosim run --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --network-mode fluid --scheduler memory_aware --output runs/custom
uv run heterosim run --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --config examples/environment.json --output runs/varying
uv run heterosim sweep --hardware profiles/hardware/uma_workloads.yaml \
  --workload examples/fork_join/workload.json --max-candidates 100 --output runs/search
uv run heterosim oracle --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --max-orders 10000 --output runs/oracle
```

`validate` 拒绝重复 ID/键、非法单位、非整数容量、NaN/Inf、断裂或不合法路由、DAG 环、数据端口与能力不匹配。输入支持 YAML/JSON；映射可以写成字典或键值列表，解析后统一为不可变 tuple。容量和 slot 使用精确整数；`32 GiB`、`100 Gbit/s`、`0.1 ms` 会显式归一化，`100G` 会被拒绝。缺省字段会出现在 resolved 配置中，哈希对同一规范化表示稳定。

成功退出码为 0；非法输入、无进展或截断退出码为 2。运行失败仍保存现场，schema/lowering 失败在进入 engine 前拒绝。

## Python API

```python
from distributed_llm_simulation import (
    Lowerer, SimulationConfig, Simulator, load_hardware,
)
from distributed_llm_simulation.workloads import SyntheticWorkload

hardware = load_hardware("profiles/hardware/pipeline_oracle.yaml")
workload, placement = SyntheticWorkload.pipeline(requests=2)
execution = Lowerer(hardware).compile(workload, placement)
config = SimulationConfig(network_mode="exclusive", scheduler="fifo")
result = Simulator(hardware, config=config).run(execution)
assert result.status == "SUCCESS"
assert abs(result.makespan_s - 0.0091) < 1e-12
```

`PlacementPlan.op_compute` 指定每个算子的设备；可用 `op_kernel` 固定 kernel。`tensor_target_pool` 指定 producer 输出或可访问输入的目标 pool，`transfer_routes` 固定某个 tensor 到目标 pool 的 route，`final_output_pools` 定义最终回传。`reload_tensor_ids` 让 offloaded tensor 每次消费时重载，释放临时设备副本；重载受当前输入和控制依赖约束。

Kernel 匹配 op、三种 dtype、可选 layout/algorithm 和 shape 范围。Roofline 对同一物理控制器的读写先求和，calibrated duration 不重复收取本地 IO 时间。`duration_s` 是固定时长覆盖值；声明的可审计 IO 不得超过该时长内的资源容量。

## 执行语义

- 静态 spec 不可变，每次 run 独立创建状态。硬件保留平行有向 channel，route 保存 channel ID；共享资源预算独立于 channel。
- UMA 多视图归一到一个物理 pool。输出与 workspace 一次原子预约；只有 VALID 副本可读，输入 pin 保留至 compute/copy 完成。扇出与进行中的搬运复用同一目标副本。
- 仅使用合法的预定义 route。没有 P2P/RDMA route 时可沿多条 route 经 RAM staging，每跳单独分配、传输和发布有效副本；无路可走明确失败。
- A 层用独占 bandwidth gate，B 层只对 transfer 做 byte-rate weighted max-min；kernel 仍为固定时长并独占声明的内存带宽组。预约包含 startup，是保守模型。
- C 层重放分段容量轨迹，D 层使用事先生成并保存的 Markov 场景。先积分旧速率、处理完成，再应用同刻变更；generation 使旧预测失效。零带宽等待未来恢复，永久停顿返回现场。
- 固定时长 kernel 的 IO 预算不能同时变化；这类配置明确拒绝。没有将 compute 伪装成已实现的流体 streaming kernel。

## 结果与研究工具

每个 run 保存 resolved hardware、workload、placement、execution、config、manifest、JSON summary、资源/请求 CSV、活动 timeline、JSONL trace 和外生轨迹。manifest 包含输入哈希、seed、版本、模型假设和源码 revision/dirty 状态。失败另存 allocation、pin、slot、等待原因和零速率活动快照。

`replay` 重新仿真并比较 summary 与 trace。指标包括请求延迟 nearest-rank 分位数、吞吐、逐 pool 峰值/byte-seconds、资源服务积分及 slot/gate 预约率、compute/transfer 时间并集和重叠。`critical_path.json` 结合实际依赖和观察到的独占资源顺序解释耗时；内存和多槽争用仍需查看 WAITING 记录。

`analysis.search` 提供缓存 evaluator、placement 枚举、向上 rank 驱动的贪心规划、内存预算扫描与经验 Pareto 集。缓存 key 包括完整 spec/config/场景/seed/版本。枚举的最佳启发式结果是可行上界。`oracle` 枚举固定 A 层执行计划的拓扑 start order，允许主动空闲；只有枚举耗尽且没有截断时才报告该范围内已证明最优。该证明固定 placement、kernel、route、容量、生命周期与变换，不是全局异构硬件极限。

`analysis.bounds` 提供有 scope/assumptions/fixed_decisions 的计算依赖全局下界、固定计划依赖/负载条件下界和容量积分界；组合取 max，并报告 gap。没有自动套用通用 IO 下界或输出未建模的功耗、费用。

## Workload 和模型边界

通用插件包括 fork-join、图像 pipeline、分块矩阵乘。Transformer 插件展开 norm、projection、attention、residual、MLP，支持 GQA、有限 prefill/decode、append-only KV、TP ring collective、PP microbatch 和显式 offload 重载。collective 模块还提供 reduce-scatter、all-gather、broadcast；消息路由、局部 reduction/pack IO 和缓冲区占用均进入通用图。KV 保留到 run 结束。LLM 模块从通用 marker 计算 TTFT、ITL、TPOT 和 token throughput。

所有自带硬件和测量数据均为 **synthetic**。校准接口支持 provenance、copy 的线性拟合和独立留出验证；还没有真实设备测量。研究扩展如动态 continuous batching、分页/共享前缀 KV、自动 eviction/recomputation、流体 compute、故障恢复和通用 IO 定理尚未实现。详见 [实现与验收状态](docs/implementation-status.md)。

## 开发验证

```sh
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

测试覆盖手算 oracle、守恒、OOM 原子回滚、别名与生命周期、staging、重放、插件、搜索与上下界标签。通过测试验证的是已声明模型的实现，不是对真实硬件精度的背书。
