import argparse
import contextlib
import datetime
import html
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from analysis.report import confirmed_verdict, feasibility, summarize_trial_group

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "org.xrgpu"
COMMANDS = (
    "inspect",
    "correctness",
    "baseline",
    "layers",
    "paired",
    "model-audit",
    "model-screen",
    "model-paired",
    "model-sustained",
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def choose_device(adb, serial=None):
    result = subprocess.run([adb, "devices", "-l"], check=True, capture_output=True, text=True)
    devices = [
        line.split()[:2]
        for line in result.stdout.splitlines()[1:]
        if line.strip() and not line.startswith("*")
    ]
    if serial:
        matches = [d for d in devices if d[0] == serial]
        if len(matches) != 1 or matches[0][1] != "device":
            raise RuntimeError(f"Device {serial} is absent or unauthorized: {devices}")
        return serial
    if len(devices) != 1 or devices[0][1] != "device":
        raise RuntimeError(
            f"Expected one authorized headset; found {devices}. Use --serial if needed."
        )
    return devices[0][0]


class Device:
    def __init__(self, adb, serial, directory, timeout=1800):
        self.adb, self.serial, self.directory, self.timeout = adb, serial, directory, timeout

    def run(self, *args, check=True):
        result = subprocess.run(
            [self.adb, "-s", self.serial, *args], capture_output=True, timeout=45
        )
        if check and result.returncode:
            raise RuntimeError(
                result.stderr.decode("utf-8", errors="replace").strip()
                or result.stdout.decode("utf-8", errors="replace").strip()
            )
        return result

    @contextlib.contextmanager
    def awake(self):
        record = {"serial": self.serial, "proximity_override": True, "restored": False}
        path = self.directory / "power-control.json"
        save_json(path, record)
        try:
            self.run("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
            yield
        finally:
            self.run(
                "shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable"
            )
            record["restored"] = True
            save_json(path, record)

    def fetch(self, remote, local):
        result = self.run("shell", "-T", "run-as", PACKAGE, "cat", "files/" + remote, check=False)
        if result.returncode:
            return None
        # validate the complete snapshot before replacing the last good local copy
        value = json.loads(result.stdout)
        local.write_bytes(result.stdout)
        return value

    def collect(self, run_id):
        return self.fetch("results/" + run_id + ".json", self.directory / (run_id + ".json"))

    def command(self, command, length=128, variant=0, profile=None):
        if command not in COMMANDS or length not in (32, 128, 256) or variant < 0:
            raise ValueError("Invalid diagnostic command")
        run_id = command + "-" + uuid.uuid4().hex
        request = {
            "run_id": run_id,
            "command": command,
            "length": length,
            "variant": variant,
            "device_serial": self.serial,
            "profile": profile,
        }
        save_json(self.directory / (run_id + "-request.json"), request)
        status_path = self.directory / (run_id + "-status.json")
        started = time.monotonic()
        print(f"{run_id}: starting (variant {variant})", flush=True)
        terminal = False
        try:
            launched = self.run(
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                PACKAGE + "/.DebugActivity",
                "--es",
                "run_id",
                run_id,
                "--es",
                "command",
                command,
                "--ei",
                "length",
                str(length),
                "--ei",
                "variant",
                str(variant),
                "--es",
                "profile",
                profile or "reference",
            )
            launch_text = (launched.stdout + launched.stderr).decode("utf-8", errors="replace")
            if any(line.startswith(("Error", "Exception")) for line in launch_text.splitlines()):
                raise RuntimeError(
                    "Android could not launch the diagnostic: " + launch_text.strip()
                )
            while time.monotonic() - started < self.timeout:
                status = self.fetch("commands/" + run_id + ".json", status_path)
                report = self.collect(run_id)
                if status:
                    if status.get("run_id") != run_id or status.get("command") != command:
                        raise RuntimeError("Mismatched command response")
                    if status.get("status") in ("complete", "failed", "canceled"):
                        terminal = True
                        if status["status"] != "complete":
                            raise RuntimeError(f"{run_id}: {status.get('error', status['status'])}")
                        if not report or report.get("status") != "complete":
                            raise RuntimeError("Command completed without a complete report")
                        print(f"{run_id}: complete", flush=True)
                        return report
                pid = self.run("shell", "pidof", PACKAGE, check=False)
                if not pid.stdout.strip() and time.monotonic() - started > 15:
                    raise RuntimeError("App exited before completing the diagnostic")
                time.sleep(5)
            raise TimeoutError(f"{run_id}: diagnostic timed out")
        except (Exception, KeyboardInterrupt) as error:
            save_json(
                self.directory / (run_id + "-host-error.json"),
                {**request, "status": "inconclusive", "error": str(error)},
            )
            raise
        finally:
            try:
                if not terminal:
                    self.run(
                        "shell",
                        "am",
                        "start",
                        "-n",
                        PACKAGE + "/.DebugActivity",
                        "--es",
                        "run_id",
                        run_id,
                        "--es",
                        "command",
                        "cancel",
                        check=False,
                    )
                self.collect(run_id)
            except (Exception, KeyboardInterrupt):
                pass 
            try:
                log = self.run(
                    "logcat", "-d", "-t", "500", "-s", "xrGPU:I", "AndroidRuntime:E", check=False
                )
                (self.directory / (run_id + ".log")).write_bytes(log.stdout + log.stderr)
            except (Exception, KeyboardInterrupt):
                pass  

    def stable(self):
        previous = None
        consecutive = 0
        for _ in range(60):
            state = self.command("inspect")
            value = (state["thermal_status"], state["charging"])
            consecutive = consecutive + 1 if value == previous and 0 <= value[0] < 3 else 0
            if consecutive >= 2:
                return
            previous = value
            time.sleep(10)
        raise RuntimeError("Headset did not reach stable thermal/charging conditions")

    def paired(self, length, variant):
        for attempt in range(3):
            self.stable()
            report = self.command("paired", length, variant)
            diagnostic = feasibility(report)
            if diagnostic["valid_evidence"]:
                return report
            thermal = [r for r in diagnostic["reasons"] if "thermal" in r or "charging" in r]
            if not thermal or attempt == 2:
                return report
            print("Discarding thermally confounded comparison; allowing stabilization.", flush=True)
        raise AssertionError("unreachable")

    def baseline(self, length):
        for _ in range(3):
            self.stable()
            report = self.command("baseline", length)
            trials = [report["cpu_baseline"], *report["trials"]]
            if all(
                t.get("status") == "ok"
                and t.get("correctness") is True
                and not t.get("thermal_discontinuity", True)
                for t in trials
            ):
                return report
            print("Baseline conditions changed; repeating after stabilization.", flush=True)
        raise RuntimeError("Could not establish a stable correct baseline")


def write_summary(directory, summary):
    save_json(directory / "summary.json", summary)
    rows = []
    for label, group in summary.get("measurements", {}).items():
        times = group["timings_ms"]

        def value(key, statistic):
            return f"{times[key][statistic]:.3f}" if times.get(key) else "unavailable"

        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in (
                    label,
                    group["correctness"],
                    group["custom_nodes"],
                    value("model_ms", "median"),
                    value("model_ms", "p95"),
                    value("end_to_end_ms", "median"),
                    value("end_to_end_ms", "p95"),
                    group.get("median_speedup", "—"),
                    group.get("paired_speedups", "—"),
                )
            )
            + "</tr>"
        )
    document = """<!doctype html><meta charset="utf-8"><title>Quest 3 feasibility</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}
table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccc;text-align:left}
pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Quest 3 MiniLM feasibility</h1>"""
    document += "<p><strong>" + html.escape(summary["verdict"]) + "</strong></p>"
    document += (
        f"<p>{summary.get('sequence_length', 128)}-token FP32 results are preliminary. GPU operation totals are not wall-clock "
        "latency. Sustained and VR acceptance have not been established.</p>"
    )
    document += (
        "<table><tr>"
        + "".join(
            "<th>" + v + "</th>"
            for v in (
                "Execution",
                "Correct",
                "Custom nodes",
                "Model median ms",
                "Model p95 ms",
                "End-to-end median ms",
                "End-to-end p95 ms",
                "Median speedup",
                "Paired speedups",
            )
        )
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )
    document += (
        "<h2>Evidence and profiling</h2><pre>"
        + html.escape(json.dumps(summary, indent=2, allow_nan=False))
        + "</pre>"
    )
    (directory / "feasibility.html").write_text(document, encoding="utf-8")


def profile_summary(profile):
    if not profile:
        return {"available": False}
    events = profile.get("gpu_events", [])
    operations = {}
    compact_events = []
    targeted_ms = 0.0
    for event in events:
        try:
            label = json.loads(event["name"])
        except (ValueError, TypeError):
            label = {}
        operator = label.get("operator", {})
        name = operator.get("name", event["name"])
        operations[name] = operations.get(name, 0.0) + event["ms"]
        args = operator.get("args", [])
        targeted = (
            name == "aten.linear.default"
            and len(args) > 1
            and args[1].get("sizes") in ([1536, 384], [384, 1536])
        )
        if targeted:
            targeted_ms += event["ms"]
        compact_events.append(
            {"operator": name, "kernel": label.get("kernel_name"), "ms": event["ms"]}
        )
    return {
        "sample": "recorded inference from trial profile_sample_index (legacy: 0); GPU totals are not critical-path latency",
        "model_ms": profile.get("model_ms"),
        "gpu_operation_total_ms": profile.get("gpu_ms"),
        "input_staging_ms": profile.get("input_staging_ms"),
        "output_staging_ms": profile.get("output_staging_ms"),
        "submit_and_wait_ms": profile.get("submit_and_wait_ms"),
        "gpu_ms_by_operator": dict(sorted(operations.items(), key=lambda item: -item[1])),
        "targeted_linear_gpu_ms": targeted_ms,
        "slowest_gpu_operations": sorted(compact_events, key=lambda e: e["ms"], reverse=True)[:8],
        "custom_linear_gpu_ms": sum(e["ms"] for e in events if "xrgpu_linear" in e["name"]),
    }


def suite(device, length):
    summary = {
        "verdict": "inconclusive",
        "device_serial": device.serial,
        "sequence_length": length,
        "measurements": {},
        "candidates": {},
        "errors": [],
        "diagnostic_only": True,
        "source_runs": {},
    }
    try:
        device.run("shell", "am", "force-stop", PACKAGE)
        caps = device.command("inspect", length)
        summary["device"] = caps
        if caps.get("device_profile") != "quest3" or not caps.get("candidate_capable"):
            raise RuntimeError("Connected device is not eligible for the Quest 3 candidates")
        baseline = device.baseline(length)
        summary["source_runs"]["baseline"] = baseline["run_id"]
        summary["measurements"]["CPU reference"] = summarize_trial_group([baseline["cpu_baseline"]])
        summary["measurements"]["Stock Vulkan baseline"] = summarize_trial_group(baseline["trials"])
        summary["baseline_profile"] = profile_summary(baseline["trials"][0].get("profile"))
        valid = []
        for variant in caps["candidate_variants"]:
            try:
                layers = device.command("layers", length, variant)
                summary["source_runs"][f"layers_{variant}"] = layers["run_id"]
                report = device.paired(length, variant)
                summary["source_runs"][f"paired_{variant}"] = report["run_id"]
                diagnostic = feasibility(report)
                summary["candidates"][str(variant)] = diagnostic
                for arm, group in diagnostic["summary"].items():
                    entry = dict(group)
                    if arm == "candidate":
                        entry.update(
                            median_speedup=diagnostic["median_speedup"],
                            paired_speedups=diagnostic["paired_speedups"],
                        )
                    summary["measurements"][f"Variant {variant}: {arm}"] = entry
                if diagnostic["valid_evidence"]:
                    valid.append((diagnostic["median_speedup"], variant, report))
            except RuntimeError as error:
                summary["errors"].append(f"Variant {variant}: {error}")
            write_summary(device.directory, summary)
        if valid:
            _, variant, first = max(valid, key=lambda entry: entry[0])
            # Start a fresh app process, retaining all app-private evidence.
            device.run("shell", "am", "force-stop", PACKAGE)
            repeat = device.paired(length, variant)
            summary["source_runs"]["confirmation"] = repeat["run_id"]
            summary["confirmation"] = feasibility(repeat)
            for arm, group in summary["confirmation"]["summary"].items():
                entry = dict(group)
                if arm == "candidate":
                    entry.update(
                        median_speedup=summary["confirmation"]["median_speedup"],
                        paired_speedups=summary["confirmation"]["paired_speedups"],
                    )
                summary["measurements"][f"Confirmation {variant}: {arm}"] = entry
            summary["best_variant"] = variant
            summary["candidate_profile"] = profile_summary(
                next(t.get("profile") for t in first["trials"] if t["arm"] == "candidate")
            )
            summary["paired_stock_profile"] = profile_summary(
                next(t.get("profile") for t in first["trials"] if t["arm"] == "stock")
            )
            summary["verdict"] = confirmed_verdict(first, repeat)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary["errors"].append(str(error))
        raise
    finally:
        write_summary(device.directory, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument(
        "--adb",
        default=shutil.which("adb")
        or str(
            ROOT
            / "artifacts/toolchain/android-sdk/platform-tools"
            / ("adb.exe" if os.name == "nt" else "adb")
        ),
    )
    parser.add_argument("--length", type=int, choices=(32, 128, 256), default=128)
    parser.add_argument("--command", choices=("suite", *COMMANDS), default="suite")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--profile", help="Exported whole-model optimization profile")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        help="Temporarily override Quest wear detection; restore the sensor on exit",
    )
    args = parser.parse_args()
    if args.command in ("layers", "paired") and args.variant <= 0:
        parser.error("--command layers/paired requires --variant with a positive candidate ID")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    directory = args.output or ROOT / "artifacts/device-results/quest3" / (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=True)
    try:
        serial = choose_device(args.adb, args.serial)
        device = Device(args.adb, serial, directory, args.timeout)
        model = device.run("shell", "getprop", "ro.product.model").stdout.decode().strip()
        if model != "Quest 3":
            raise RuntimeError(f"Expected Quest 3, found {model}")
        with device.awake() if args.keep_awake else contextlib.nullcontext():
            if args.command == "suite":
                result = suite(device, args.length)
                print(result["verdict"])
            else:
                device.command(args.command, args.length, args.variant, args.profile)
        print(f"Evidence: {directory}")
    except (Exception, KeyboardInterrupt) as error:
        save_json(directory / "host-error.json", {"verdict": "inconclusive", "error": str(error)})
        if not (directory / "summary.json").exists():
            write_summary(
                directory,
                {
                    "verdict": "inconclusive",
                    "sequence_length": args.length,
                    "diagnostic_only": True,
                    "measurements": {},
                    "errors": [str(error)],
                },
            )
        raise SystemExit(f"No conclusion: {error}\nEvidence: {directory}") from error


if __name__ == "__main__":
    main()
