import argparse
import json
import time
from pathlib import Path

from analysis.optimization import compare_models
from analysis.report import summarize_trial_group
from runtime.device_benchmark import Device, ROOT, choose_device, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--length", type=int, choices=(32, 128, 256), default=128)
    parser.add_argument(
        "--phase", choices=("screen", "compare", "sustained", "audit"), default="screen"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()
    import contextlib
    import os

    adb = (
        ROOT
        / "artifacts/toolchain/android-sdk/platform-tools"
        / ("adb.exe" if os.name == "nt" else "adb")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    serial = choose_device(str(adb), args.serial)
    device = Device(str(adb), serial, args.output, timeout=2400)
    if device.run("shell", "getprop", "ro.product.model").stdout.decode().strip() != "Quest 3":
        raise RuntimeError("This optimization experiment targets Quest 3")
    bundle = json.loads((ROOT / "artifacts/bundle/bundle.json").read_text())
    summary = {
        "phase": args.phase,
        "sequence_length": args.length,
        "device_serial": serial,
        "variant": args.variant,
        "results": {},
        "errors": {},
    }
    command = {
        "screen": "model-screen",
        "compare": "model-paired",
        "sustained": "model-sustained",
        "audit": "model-audit",
    }[args.phase]
    try:
        with device.awake() if args.keep_awake else contextlib.nullcontext():
            device.run("shell", "am", "force-stop", "org.xrgpu")
            for profile in args.profiles:
                try:
                    spec = bundle["optimization_profiles"][profile]
                    file = spec["lengths"][str(args.length)]["model"]
                    for attempt in range(3):
                        device.stable()
                        report = device.command(command, args.length, args.variant, profile)
                        if (
                            report["candidate_sha256"] != bundle["files"][file]
                            or report["candidate_precision"] != spec["precision"]
                        ):
                            raise RuntimeError("Installed candidate does not match the host export")
                        if args.phase in ("compare", "sustained"):
                            analysis = compare_models(report, args.phase == "sustained")
                            unstable = any(
                                "thermal" in r or "charging" in r for r in analysis["reasons"]
                            )
                        else:
                            unstable = any(
                                t.get("thermal_discontinuity", True) for t in report["trials"]
                            )
                            analysis = {
                                "valid_evidence": not unstable,
                                "selection_eligible": False,
                                "summary": summarize_trial_group(report["trials"]),
                            }
                        if not unstable or attempt == 2:
                            break
                        print(
                            "Discarding confounded run and waiting for stable conditions",
                            flush=True,
                        )
                        time.sleep(10)
                    summary["results"][profile] = {"run_id": report["run_id"], **analysis}
                    print(profile, json.dumps(analysis), flush=True)
                except (RuntimeError, TimeoutError) as error:
                    summary["errors"][profile] = str(error)
                    print(f"Rejected {profile}: {error}", flush=True)
                finally:
                    save_json(args.output / "optimization-summary.json", summary)
    finally:
        save_json(args.output / "optimization-summary.json", summary)


if __name__ == "__main__":
    main()
