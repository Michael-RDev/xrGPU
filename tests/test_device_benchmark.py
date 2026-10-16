import json
import subprocess
from types import SimpleNamespace

import pytest

from runtime.device_benchmark import Device, choose_device, main, profile_summary


def test_device_discovery_rejects_unauthorized_or_ambiguous(monkeypatch):
    response = SimpleNamespace(stdout="List of devices attached\nquest unauthorized\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: response)
    with pytest.raises(RuntimeError, match="authorized"):
        choose_device("adb")
    response.stdout = "List of devices attached\nquest device\nphone device\n"
    with pytest.raises(RuntimeError, match="one authorized"):
        choose_device("adb")
    assert choose_device("adb", "quest") == "quest"


class FakeDevice(Device):
    def __init__(self, directory, status, report):
        super().__init__("adb", "quest", directory)
        self.status, self.report = status, report

    def run(self, *args, check=True):
        data = b""
        if args[:2] == ("shell", "-T"):
            value = self.status if "commands/" in args[-1] else self.report
            data = json.dumps(value).encode()
        return SimpleNamespace(returncode=0, stdout=data, stderr=b"")


@pytest.mark.parametrize(
    "status,report,error",
    [
        (
            {"status": "failed", "error": "shader failed"},
            {"status": "failed", "samples": [1]},
            "shader failed",
        ),
        ({"status": "complete"}, {"status": "running", "samples": [1]}, "complete report"),
        ({"status": "complete", "run_id": "different"}, {"status": "complete"}, "Mismatched"),
    ],
)
def test_failed_and_mismatched_commands_preserve_evidence(
    tmp_path, monkeypatch, status, report, error
):
    monkeypatch.setattr("runtime.device_benchmark.uuid.uuid4", lambda: SimpleNamespace(hex="fixed"))
    status = {"run_id": "paired-fixed", "command": "paired", **status}
    device = FakeDevice(tmp_path, status, report)
    with pytest.raises(RuntimeError, match=error):
        device.command("paired", variant=1)
    assert json.loads((tmp_path / "paired-fixed.json").read_text()) == report
    assert (tmp_path / "paired-fixed-host-error.json").exists()


def test_command_retrieves_exact_report(tmp_path, monkeypatch):
    monkeypatch.setattr("runtime.device_benchmark.uuid.uuid4", lambda: SimpleNamespace(hex="fixed"))
    device = FakeDevice(
        tmp_path,
        {"run_id": "inspect-fixed", "command": "inspect", "status": "complete"},
        {"status": "complete", "device_profile": "quest3"},
    )
    assert device.command("inspect")["device_profile"] == "quest3"


def test_truncated_fetch_keeps_last_valid_snapshot(tmp_path):
    device = Device("adb", "quest", tmp_path)
    saved = tmp_path / "report.json"
    saved.write_text('{"status":"running"}')
    device.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'{"status":')
    with pytest.raises(json.JSONDecodeError):
        device.fetch("results/run.json", saved)
    assert json.loads(saved.read_text())["status"] == "running"


def test_not_yet_created_report_uses_remote_exit_status(tmp_path):
    device = Device("adb", "quest", tmp_path)
    calls = []

    def missing(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout=b"cat: No such file or directory")

    device.run = missing
    assert device.fetch("results/run.json", tmp_path / "report.json") is None
    assert calls[0][:2] == ("shell", "-T")
    assert not (tmp_path / "report.json").exists()


def test_keep_awake_restores_sensor_after_failure(tmp_path):
    device = Device("adb", "quest", tmp_path)
    calls = []
    device.run = lambda *a: calls.append(a)
    with pytest.raises(RuntimeError, match="interrupted"):
        with device.awake():
            raise RuntimeError("interrupted")
    assert calls[0][-1] == "com.oculus.vrpowermanager.prox_close"
    assert calls[-1][-1] == "com.oculus.vrpowermanager.automation_disable"
    assert json.loads((tmp_path / "power-control.json").read_text())["restored"] is True


def test_android_launch_error_is_detected_even_with_success_exit_code(tmp_path):
    device = Device("adb", "quest", tmp_path)
    device.run = lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=b"Error type 3\nError: Activity class does not exist", stderr=b""
    )
    with pytest.raises(RuntimeError, match="could not launch"):
        device.command("inspect")
    assert len(list(tmp_path.glob("*-host-error.json"))) == 1
    assert len(list(tmp_path.glob("*.log"))) == 1


def test_profile_separates_targeted_gpu_operations_from_wall_time():
    label = json.dumps(
        {
            "operator": {"name": "aten.linear.default", "args": [{}, {"sizes": [384, 1536]}]},
            "kernel_name": "xrgpu_linear_test",
        }
    )
    result = profile_summary(
        {
            "model_ms": 20,
            "gpu_ms": 8,
            "gpu_events": [{"name": label, "ms": 3}, {"name": "other", "ms": 5}],
        }
    )
    assert result["model_ms"] == 20
    assert result["gpu_operation_total_ms"] == 8
    assert result["targeted_linear_gpu_ms"] == result["custom_linear_gpu_ms"] == 3
    assert result["gpu_ms_by_operator"]["aten.linear.default"] == 3


def test_baseline_retries_confounded_run(tmp_path):
    device = Device("adb", "quest", tmp_path)
    calls = []
    device.stable = lambda: calls.append("stable")
    good = {"status": "ok", "correctness": True, "thermal_discontinuity": False}
    reports = iter(
        [
            {"cpu_baseline": good, "trials": [{**good, "thermal_discontinuity": True}]},
            {"cpu_baseline": good, "trials": [good]},
        ]
    )
    device.command = lambda *args: next(reports)
    assert not device.baseline(128)["trials"][0]["thermal_discontinuity"]
    assert calls == ["stable", "stable"]


def test_no_device_creates_inconclusive_report_without_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark", "--adb", "adb", "--output", str(tmp_path)])
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="List of devices attached\n")
    )
    with pytest.raises(SystemExit, match="No conclusion"):
        main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict"] == "inconclusive" and summary["measurements"] == {}
    assert (tmp_path / "feasibility.html").exists()


@pytest.mark.parametrize("clock,error", [([0, 20, 21], "App exited"), ([0, 1801], "timed out")])
def test_crash_or_timeout_is_inconclusive_and_keeps_partial_data(
    tmp_path, monkeypatch, clock, error
):
    monkeypatch.setattr("runtime.device_benchmark.uuid.uuid4", lambda: SimpleNamespace(hex="fixed"))
    ticks = iter(clock)
    monkeypatch.setattr("runtime.device_benchmark.time.monotonic", lambda: next(ticks))
    device = FakeDevice(
        tmp_path,
        {"run_id": "paired-fixed", "command": "paired", "status": "running"},
        {"status": "running", "trials": [], "current_trial": {"samples": [1, 2]}},
    )
    with pytest.raises((RuntimeError, TimeoutError), match=error):
        device.command("paired", variant=1)
    partial = json.loads((tmp_path / "paired-fixed.json").read_text())
    assert partial["current_trial"]["samples"] == [1, 2]
    assert (
        json.loads((tmp_path / "paired-fixed-host-error.json").read_text())["status"]
        == "inconclusive"
    )
