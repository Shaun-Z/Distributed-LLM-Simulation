import json
import subprocess
import sys

import pytest

from distributed_llm_simulation.cli import main


def test_demo_export_inspect_replay_and_run(tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["demo", "pipeline", "--output", str(output)]) == 0
    expected = {
        "hardware.resolved.yaml",
        "workload.json",
        "placement.json",
        "config.json",
        "manifest.json",
        "summary.json",
        "resources.csv",
        "requests.csv",
        "trace.jsonl",
        "environment_trace.jsonl",
        "execution.json",
        "timeline.json",
        "critical_path.json",
        "bounds.json",
    }
    assert expected <= {p.name for p in output.iterdir()}
    assert json.loads((output / "summary.json").read_text())["makespan_s"] == pytest.approx(0.0091)
    assert main(["replay", str(output)]) == 0
    assert main(["inspect", str(output), "--timeline"]) == 0
    assert (
        main(
            [
                "validate",
                "--hardware",
                str(output / "hardware.resolved.yaml"),
                "--workload",
                str(output / "workload.json"),
                "--placement",
                str(output / "placement.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run",
                "--hardware",
                str(output / "hardware.resolved.yaml"),
                "--workload",
                str(output / "workload.json"),
                "--placement",
                str(output / "placement.json"),
                "--output",
                str(tmp_path / "run2"),
            ]
        )
        == 0
    )
    assert main(["demo", "pipeline", "--output", str(output)]) == 2
    assert "nonempty" in capsys.readouterr().out


def test_sweep_and_budget(tmp_path):
    output = tmp_path / "inputs"
    assert main(["demo", "matmul", "--output", str(output)]) == 0
    args = [
        "sweep",
        "--hardware",
        str(output / "hardware.resolved.yaml"),
        "--workload",
        str(output / "workload.json"),
    ]
    assert main([*args, "--max-candidates", "2", "--output", str(tmp_path / "sweep")]) == 0
    report = json.loads((tmp_path / "sweep/sweep.json").read_text())
    assert report["candidates_evaluated"] == 2 and report["exhausted"] is False
    budgets = tmp_path / "budgets.json"
    budgets.write_text('[{"ram_capacity":0},{"ram_capacity":1000000000}]')
    assert (
        main(
            [
                *args,
                "--budgets",
                str(budgets),
                "--placement",
                str(output / "placement.json"),
                "--output",
                str(tmp_path / "budget"),
            ]
        )
        == 0
    )
    rows = json.loads((tmp_path / "budget/sweep.json").read_text())["evaluations"]
    assert rows[0]["status"] == "NO_PROGRESS" and rows[1]["status"] == "SUCCESS"


def test_failure_exit_and_snapshot(tmp_path):
    output = tmp_path / "inputs"
    main(["demo", "pipeline", "--output", str(output)])
    hardware = output / "hardware.resolved.yaml"
    import yaml

    data = yaml.safe_load(hardware.read_text())
    next(r for r in data["resources"] if r["id"] == "ram_capacity")["capacity"] = 0
    hardware.write_text(yaml.safe_dump(data))
    failure = tmp_path / "failed"
    command = [
        sys.executable,
        "-m",
        "distributed_llm_simulation",
        "run",
        "--hardware",
        str(hardware),
        "--workload",
        str(output / "workload.json"),
        "--placement",
        str(output / "placement.json"),
        "--output",
        str(failure),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads((failure / "summary.json").read_text())["status"] == "NO_PROGRESS"
    assert (failure / "failure_snapshot.json").is_file()


def test_oracle_winner_replays(tmp_path):
    inputs = tmp_path / "inputs"
    assert main(["demo", "pipeline", "--requests", "1", "--output", str(inputs)]) == 0
    output = tmp_path / "oracle"
    assert (
        main(
            [
                "oracle",
                "--hardware",
                str(inputs / "hardware.resolved.yaml"),
                "--workload",
                str(inputs / "workload.json"),
                "--placement",
                str(inputs / "placement.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "oracle.json").read_text())
    assert report["proven_optimal_in_scope"] and report["incumbent_s"] == pytest.approx(0.0061)
    assert main(["replay", str(output / "best")]) == 0


def test_overwrite_removes_obsolete_optional_reports(tmp_path):
    output = tmp_path / "run"
    assert main(["demo", "transformer", "--output", str(output)]) == 0
    assert (output / "llm_metrics.json").is_file() and (output / "bounds.json").is_file()
    assert (
        main(
            [
                "demo",
                "pipeline",
                "--config",
                "examples/environment.json",
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        == 0
    )
    assert not (output / "llm_metrics.json").exists() and not (output / "bounds.json").exists()
    assert main(["replay", str(output)]) == 0


def test_budget_points_keep_auditable_runs(tmp_path):
    output = tmp_path / "budget"
    assert (
        main(
            [
                "sweep",
                "--hardware",
                "profiles/hardware/pipeline_oracle.yaml",
                "--workload",
                "examples/pipeline/workload.json",
                "--placement",
                "examples/pipeline/placement.json",
                "--budgets",
                "examples/memory-budgets.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "point-0000/failure_snapshot.json").is_file()
    assert main(["replay", str(output / "point-0002")]) == 0


def test_replay_detects_semantically_invisible_input_tampering(tmp_path):
    output = tmp_path / "run"
    assert main(["demo", "pipeline", "--output", str(output)]) == 0
    import yaml

    path = output / "hardware.resolved.yaml"
    data = yaml.safe_load(path.read_text())
    data["name"] += "-edited"
    path.write_text(yaml.safe_dump(data))
    assert main(["replay", str(output)]) == 2
