import math


def nearest_rank(values, quantile):
    if not 0 < quantile <= 1:
        raise ValueError("quantile must lie in (0, 1]")
    return sorted(values)[math.ceil(quantile * len(values)) - 1] if values else None


def interval_union(intervals):
    end, total = -math.inf, 0.0
    for start, finish in sorted(intervals):
        total += max(0.0, finish - max(start, end))
        end = max(end, finish)
    return total


def replication_statistics(results):
    """Intervals use independent run means, never correlated requests as replicates."""
    import statistics

    successful = [r.makespan_s for r in results if r.status == "SUCCESS"]
    n = len(successful)
    mean = statistics.mean(successful) if n else None
    se = statistics.stdev(successful) / math.sqrt(n) if n > 1 else None
    return {
        "replications": len(results),
        "successful_replications": n,
        "failure_rate": 1 - n / len(results) if results else None,
        "mean_success_makespan_s": mean,
        "standard_error": se,
        "normal_approx_95_interval": [mean - 1.96 * se, mean + 1.96 * se]
        if se is not None
        else None,
        "interval_assumptions": "independent replications; normal approximation; small samples and tails require more data",
        "p95_success_makespan_s": nearest_rank(successful, 0.95),
    }
