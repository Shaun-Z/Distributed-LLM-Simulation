# 可重复实验

所有命令在仓库根目录执行。`runs/` 已忽略，不提交运行产生的大量 trace。

```sh
uv run heterosim run --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --config examples/environment.json --output runs/pipeline-varying
uv run heterosim replay runs/pipeline-varying
uv run python examples/research.py --output runs/tp-pp-replications
uv run heterosim sweep --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --budgets examples/memory-budgets.json --output runs/memory-sweep
```

`research.py` 为 TP/PP 使用相同 seed 的相同外生网络轨迹，按独立 run 计算均值、标准误和正态近似区间。三个 seed 只是用法演示，不能据此宣称稳定 P99。固定轨迹标记 C，预生成 Markov 场景标记 D；查询容量不会抽样。配置保存全过程，`replay` 无需重新采样。

汇报需说明：workload 粒度、硬件是否 synthetic、A/B/C/D、路由和 output contract、空间向量、策略、场景数量、失败率、条件下界及假设。`bounds.json` 的目标是从 t=0 起的 finish time；请求窗口 makespan 和 arrival 可能不同。
