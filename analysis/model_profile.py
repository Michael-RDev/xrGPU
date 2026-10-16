import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarize_profiles(trials):
    result = {}
    for arm in sorted({t["arm"] for t in trials}):
        selected = [t for t in trials if t["arm"] == arm and t.get("status") == "ok"]
        ops, dispatches, wall, gpu, staging, waiting = defaultdict(list), [], [], [], [], []
        for trial in selected:
            profile = trial.get("profile", {})
            totals = defaultdict(float)
            events = profile.get("gpu_events", [])
            for event in events:
                try:
                    name = json.loads(event["name"])["operator"]["name"]
                except (ValueError, KeyError, TypeError):
                    name = "unattributed"
                totals[name] += event["ms"]
            for name, ms in totals.items():
                ops[name].append(ms)
            if events:
                dispatches.append(len(events))
                gpu.append(sum(e["ms"] for e in events))
            for sample in trial.get("samples", []):
                wall.append(sample["model_ms"])
                if (
                    sample.get("input_staging_ms") is not None
                    and sample.get("output_staging_ms") is not None
                ):
                    staging.append(sample["input_staging_ms"] + sample["output_staging_ms"])
                if sample.get("submit_and_wait_ms") is not None:
                    waiting.append(sample["submit_and_wait_ms"])
        result[arm] = {
            "model_wall_median_ms": median(wall) if wall else None,
            "staging_median_ms": median(staging) if staging else None,
            "submit_and_wait_median_ms": median(waiting) if waiting else None,
            "profiled_gpu_event_sum_median_ms": median(gpu) if gpu else None,
            "profiled_dispatch_count_median": median(dispatches) if dispatches else None,
            "operators": dict(
                sorted(((k, median(v)) for k, v in ops.items()), key=lambda p: -p[1])
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_profiles(json.loads(args.report.read_text())["trials"]), indent=2))


if __name__ == "__main__":
    main()
