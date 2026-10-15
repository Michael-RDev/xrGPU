package org.xrgpu;

import android.content.Context;
import android.os.*;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public final class BenchmarkRunner {
    final Context context;
    final Assets assets;
    final JSONObject caps;
    final PowerManager power;
    final AtomicBoolean cancel;
    final Consumer<String> progress;
    final int warmups, iterations, pairs, intervalMs, sustainedSeconds;
    public BenchmarkRunner(Context context, Assets assets, AtomicBoolean cancel, Consumer<String> progress) throws Exception {
        this.context = context; this.assets = assets; this.cancel = cancel; this.progress = progress;
        caps = new JSONObject(Native.capabilities()); power = context.getSystemService(PowerManager.class);
        JSONObject suite = new JSONObject(assets.text("suite.json"));
        warmups=suite.getInt("warmup_iterations"); iterations=suite.getInt("measured_iterations");
        pairs=suite.getInt("paired_trials"); intervalMs=suite.getInt("inference_interval_ms");
        sustainedSeconds=suite.getInt("sustained_seconds");
        if(warmups<20 || iterations<100 || pairs<4 || intervalMs<1 || sustainedSeconds<600)
            throw new IllegalArgumentException("Suite is below the minimum evidence requirements");
    }
    void check(boolean vr) throws Exception {
        if (cancel.get() || Thread.currentThread().isInterrupted()) throw new InterruptedException("Run canceled");
        if (power.getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_SEVERE)
            throw new IllegalStateException("Thermal state is severe; run stopped");
        if (vr) {
            JSONObject s = new JSONObject(Native.sceneStatus());
            if (!s.optBoolean("focused") || Math.abs(s.optDouble("refresh_hz", 0) - 72) > .1)
                throw new IllegalStateException("VR scene lost focus or is not running at 72 Hz");
        }
    }
    JSONObject metadata(int length, boolean vr) throws Exception {
        JSONObject manifest = assets.manifest;
        JSONObject out = new JSONObject().put("device", Build.MANUFACTURER + "/" + Build.MODEL)
            .put("device_profile", caps.get("device_profile")).put("kernel_set", caps.get("kernel_set"))
            .put("candidate_variants", caps.getJSONArray("candidate_variants"))
            .put("gpu", caps.opt("gpu")).put("vendor_id", caps.opt("vendor_id"))
            .put("device_id", caps.opt("device_id")).put("api_version", caps.opt("api_version"))
            .put("driver", caps.opt("driver")).put("os", Build.FINGERPRINT)
            .put("model_revision", manifest.get("model_revision"))
            .put("model_sha256", manifest.getJSONObject("files").get("minilm_vulkan_" + length + ".pte"))
            .put("runtime_revision", caps.get("runtime_revision")).put("kernel_revision", caps.get("kernel_revision"))
            .put("precision", "fp32").put("sequence_length", length).put("mode", vr ? "vr" : "isolated")
            .put("scene_revision", vr ? "cube-grid-v1-1440x1440-gles3" : "none")
            .put("refresh_hz", vr ? new JSONObject(Native.sceneStatus()).optDouble("refresh_hz", 0) : 0)
            .put("inference_interval_ms", intervalMs).put("suite_revision", manifest.get("suite_revision"))
            .put("charging", context.getSystemService(BatteryManager.class).isCharging());
        return out;
    }
    File selectionFile(int length, boolean vr) throws Exception {
        return new File(context.getFilesDir(), "selections/" + caps.getString("device_profile") +
            "/" + length + "-" + (vr ? "vr" : "isolated") + ".json");
    }
    JSONObject trial(int pair, int variant, int length, boolean vr, double seconds, String backend) throws Exception {
        check(vr);
        JSONObject out = metadata(length, vr).put("pair", pair).put("arm", variant == 0 ? "stock" : "candidate")
            .put("variant", variant).put("backend", backend).put("warmup_iterations", warmups);
        if (backend.equals("cpu")) out.put("model_sha256", assets.manifest.getJSONObject("files").get("minilm_cpu_" + length + ".pte"));
        JSONArray samples = new JSONArray();
        try (Native.Model model = new Native.Model(assets.model(backend, length), variant)) {
            Validation.embeddings(assets, model, length);
            String text = "Find the flashlight so I can see in the dark virtual room.";
            for (int i = 0; i < warmups; i++) { check(vr); model.encode(assets.tokenizer.encode(text, length)); }
            int initialThermal = power.getCurrentThermalStatus(); boolean thermalChanged = false;
            out.put("thermal_start", initialThermal);
            if (vr) Native.sceneStats(true);
            long started = SystemClock.elapsedRealtimeNanos();
            do {
                check(vr); long start = SystemClock.elapsedRealtimeNanos();
                model.encode(assets.tokenizer.encode(text, length));
                double elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1e6;
                JSONObject stats = new JSONObject(model.stats());
                int thermal = power.getCurrentThermalStatus(); thermalChanged |= thermal != initialThermal;
                boolean charging=context.getSystemService(BatteryManager.class).isCharging();
                thermalChanged |= charging != out.getBoolean("charging");
                JSONObject sample = new JSONObject().put("end_to_end_ms", elapsed)
                    .put("model_ms", stats.get("model_ms")).put("gpu_ms", stats.get("gpu_ms"))
                    .put("input_staging_ms", stats.get("input_staging_ms"))
                    .put("output_staging_ms", stats.get("output_staging_ms"))
                    .put("submit_and_wait_ms", stats.get("submit_and_wait_ms"))
                    .put("thermal_status", thermal).put("charging", charging).put("process_pss_kb", Debug.getPss())
                    .put("elapsed_seconds", (SystemClock.elapsedRealtimeNanos() - started) / 1e9);
                samples.put(sample);
                if (samples.length() == 1) out.put("profile", stats);
                // Pace from the start of each inference. No catch-up bursts.
                long remaining = intervalMs * 1_000_000L - (SystemClock.elapsedRealtimeNanos() - start);
                if (remaining > 0) Thread.sleep(remaining / 1_000_000L, (int)(remaining % 1_000_000L));
            } while (samples.length() < iterations || (SystemClock.elapsedRealtimeNanos() - started) / 1e9 < seconds);
            out.put("measurement_seconds", (SystemClock.elapsedRealtimeNanos() - started) / 1e9);
            if (vr) {
                JSONObject frames = new JSONObject(Native.sceneStats(false));
                out.put("frame_ms", frames.getJSONArray("frame_ms"))
                    .put("missed_frame_rate", frames.opt("missed_frame_rate"))
                    .put("frame_metrics", frames);
            }
            Validation.embeddings(assets, model, length);
            JSONObject stats = new JSONObject(model.stats());
            out.put("samples", samples).put("custom_nodes", stats.getLong("custom_nodes"))
                .put("cold_load_ms", stats.getDouble("load_ms")).put("correctness", true)
                .put("thermal_discontinuity", thermalChanged).put("status", "ok");
        }
        return out;
    }
    public File paired(int length, int variant, boolean vr, boolean sustained) throws Exception {
        if (!Acceptance.allowsVariant(caps, variant)) throw new IllegalArgumentException("No such candidate in this device profile");
        JSONObject report = new JSONObject().put("schema_version", 1).put("trials", new JSONArray());
        File output = new File(context.getFilesDir(), "results/" + System.currentTimeMillis() + "-" + (vr ? "vr" : "isolated") + ".json");
        try {
            progress.accept("Checking CPU reference...");
            report.put("cpu_baseline", trial(-1, 0, length, vr, 0, "cpu"));
            for (int pair = 0; pair < pairs; pair++) {
                int[] order = pair % 2 == 0 ? new int[]{0, variant} : new int[]{variant, 0};
                for (int v : order) {
                    progress.accept("Pair " + (pair + 1) + "/" + pairs + " — " + (v == 0 ? "stock" : "candidate " + v));
                    report.getJSONArray("trials").put(trial(pair, v, length, vr, 0, "vulkan"));
                    Assets.writeAtomic(output, report);
                }
            }
            if (sustained) {
                JSONObject longRun = new JSONObject().put("schema_version",1).put("trials", new JSONArray()); double duration = 0;
                for (int pair = 0; pair < 5; pair++) {
                    int[] order = pair % 2 == 0 ? new int[]{0, variant} : new int[]{variant, 0};
                    for (int v : order) {
                        progress.accept("Sustained comparison " + (pair + 1) + "/5 — " + (v == 0 ? "stock" : "candidate"));
                        JSONObject t = trial(pair, v, length, vr, sustainedSeconds / 10.0, "vulkan");
                        longRun.getJSONArray("trials").put(t); duration += t.getDouble("measurement_seconds");
                        longRun.put("duration_seconds", duration); report.put("sustained", longRun);
                        Assets.writeAtomic(output, report);
                    }
                }
            }
            report.put("acceptance", Acceptance.evaluate(report, true)).put("status", "complete");
            Assets.writeAtomic(output, report);
            if (report.getJSONObject("acceptance").getBoolean("accepted")) {
                // Only on-device, evidence-backed profiles are eligible for selection.
                JSONObject selection = metadata(length, vr).put("variant", variant)
                    .put("report", output.getName()).put("report_sha256", Assets.digest(output));
                Assets.writeAtomic(selectionFile(length, vr), selection);
            }
            progress.accept(report.getJSONObject("acceptance").toString(2));
            return output;
        } catch (Exception e) {
            report.put("status", "failed").put("error", e.toString()); Assets.writeAtomic(output, report); throw e;
        }
    }
    public File layers(int variant) throws Exception {
        if (!Acceptance.allowsVariant(caps, variant)) throw new IllegalArgumentException("No such candidate in this device profile");
        JSONArray cases = new JSONArray(assets.text("layers.json")), results = new JSONArray();
        File output = new File(context.getFilesDir(), "results/" + System.currentTimeMillis() + "-layers.json");
        JSONObject report = new JSONObject().put("schema_version", 1).put("kind", "layers")
            .put("capabilities", caps).put("precision", "fp32").put("model_revision", assets.manifest.get("model_revision"))
            .put("suite_revision", assets.manifest.get("suite_revision")).put("results", results);
        try {
        for (int i = 0; i < cases.length(); i++) {
            check(false); JSONObject c = cases.getJSONObject(i); String id = c.getString("id");
            int rows = c.getJSONArray("shape").getInt(1), cols = c.getJSONArray("shape").getInt(2);
            float[] input = assets.floats(id + ".input.bin"), expected = assets.floats(id + ".expected.bin");
            for (int v : new int[]{0, variant}) {
                progress.accept(id + " — variant " + v);
                try (Native.Model model = new Native.Model(new File(assets.root, id + ".pte").getPath(), v)) {
                    for (int warm = 0; warm < 20; warm++) { check(false); Validation.layer(model.layer(input, rows, cols), expected); }
                    JSONArray samples = new JSONArray();
                    for (int sample = 0; sample < 100; sample++) {
                        check(false); float[] actual = model.layer(input, rows, cols);
                        samples.put(new JSONObject(model.stats())); Validation.layer(actual, expected);
                    }
                    long nodes = new JSONObject(model.stats()).getLong("custom_nodes");
                    boolean shouldDispatch = model.supportsLinear(rows, cols, c.getJSONArray("output_shape").getInt(2));
                    if ((nodes > 0) != shouldDispatch) throw new IllegalStateException("Unexpected kernel dispatch for " + id);
                    results.put(new JSONObject().put("case", id).put("variant", v).put("correctness", true)
                        .put("shape", c.get("shape")).put("output_shape", c.get("output_shape"))
                        .put("program_sha256", assets.manifest.getJSONObject("files").get(id + ".pte"))
                        .put("samples", samples));
                    Assets.writeAtomic(output, report);
                }
            }
        }
        report.put("status", "complete"); Assets.writeAtomic(output, report); return output;
        } catch (Exception e) {
            report.put("status", "failed").put("error", e.toString()); Assets.writeAtomic(output, report); throw e;
        }
    }
}
