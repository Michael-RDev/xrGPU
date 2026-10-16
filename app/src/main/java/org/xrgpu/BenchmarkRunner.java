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
    File commandOutput;
    private File checkpointOutput;
    private JSONObject checkpointReport;

    private File output(String kind) {
        return commandOutput != null ? commandOutput : new File(context.getFilesDir(),
            "results/" + System.currentTimeMillis() + "-" + kind + ".json");
    }
    private void beginReport(File output, JSONObject report) throws Exception {
        checkpointOutput = output; checkpointReport = report;
        if (commandOutput != null) report.put("run_id", commandOutput.getName().replace(".json", ""));
        report.put("status", "running"); Assets.writeAtomic(output, report);
    }
    private void checkpoint(JSONObject trial) throws Exception {
        if (checkpointReport != null) {
            checkpointReport.put("current_trial", trial);
            Assets.writeAtomic(checkpointOutput, checkpointReport);
        }
    }
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
    File modelSelectionFile(int length) throws Exception {
        return new File(context.getFilesDir(), "selections/"+caps.getString("device_profile")+"/model-"+length+"-isolated.json");
    }
    JSONObject trial(int pair, int variant, int length, boolean vr, double seconds, String backend) throws Exception {
        return trial(pair,variant,length,vr,seconds,backend,null,variant==0?"stock":"candidate");
    }
    JSONObject trial(int pair, int variant, int length, boolean vr, double seconds, String backend,
                     String profile, String arm) throws Exception {
        check(vr);
        JSONObject out = metadata(length, vr).put("pair", pair).put("arm", arm)
            .put("variant", variant).put("backend", backend).put("warmup_iterations", warmups);
        if (backend.equals("cpu")) out.put("model_sha256", assets.manifest.getJSONObject("files").get("minilm_cpu_" + length + ".pte"));
        String path=assets.model(backend,length);
        out.put("model_profile",profile==null?"reference":profile);
        if(profile!=null) {
            JSONObject spec=optimizationProfile(profile,length);
            String file=spec.getJSONObject("lengths").getJSONObject(Integer.toString(length)).getString("model");
            path=new File(assets.root,file).getPath();
            out.put("model_sha256",assets.manifest.getJSONObject("files").get(file)).put("precision",spec.getString("precision"));
        }
        JSONArray samples = new JSONArray();
        out.put("samples", samples).put("status", "running");
        checkpoint(out);
        try (Native.Model model = new Native.Model(path, variant)) {
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
                if (samples.length() == 1 || samples.length() == 50)
                    out.put("profile", stats).put("profile_sample_index",samples.length()-1);
                if (samples.length() % 25 == 0) checkpoint(out);
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
    JSONObject optimizationProfile(String name,int length) throws Exception {
        JSONObject spec=assets.manifest.getJSONObject("optimization_profiles").getJSONObject(name);
        spec.getJSONObject("lengths").getJSONObject(Integer.toString(length));
        return spec;
    }
    JSONObject auditModel(String name,int length,int variant) throws Exception {
        JSONObject spec=optimizationProfile(name,length).getJSONObject("lengths").getJSONObject(Integer.toString(length));
        JSONArray cases=spec.getJSONArray("audit_cases"), layout=spec.getJSONArray("checkpoints"), results=new JSONArray();
        JSONObject result=new JSONObject().put("correctness",true).put("atol",1e-4).put("rtol",1e-4)
            .put("checkpoints",layout.length()).put("cases",results)
            .put("audit_model_sha256",assets.manifest.getJSONObject("files").get(spec.getString("audit_model")));
        try(Native.Model model=new Native.Model(new File(assets.root,spec.getString("audit_model")).getPath(),variant)) {
            for(int c=0;c<cases.length();c++) {
                check(false); JSONObject test=cases.getJSONObject(c);
                float[] expected=assets.floats(test.getString("expected"));
                float[] actual=model.encode(assets.tokenizer.encode(test.getString("text"),length));
                if(actual.length!=expected.length) throw new IllegalStateException("Checkpoint output shape mismatch");
                JSONArray checks=new JSONArray();
                for(int j=0;j<layout.length();j++) {
                    JSONObject point=layout.getJSONObject(j); double max=0, ratio=0; boolean ok=true;
                    int offset=point.getInt("offset"), count=point.getInt("count");
                    for(int k=offset;k<offset+count;k++) {
                        double error=Math.abs(actual[k]-(double)expected[k]);
                        if(!Float.isFinite(actual[k])) { ok=false; continue; }
                        max=Math.max(max,error); ratio=Math.max(ratio,error/(1e-4+1e-4*Math.abs(expected[k])));
                        if(error>1e-4+1e-4*Math.abs(expected[k])) ok=false;
                    }
                    checks.put(new JSONObject().put("name",point.getString("name")).put("correctness",ok)
                        .put("max_abs_error",max).put("max_tolerance_ratio",ratio));
                    if(!ok) result.put("correctness",false);
                }
                results.put(new JSONObject().put("text",test.getString("text")).put("checks",checks));
                progress.accept("Checkpoint audit "+name+": "+(c+1)+"/"+cases.length());
            }
        }
        return result;
    }
    public File modelExperiment(String command,String name,int length,int variant) throws Exception {
        JSONObject spec=optimizationProfile(name,length);
        if(variant!=0 && (!Acceptance.allowsVariant(caps,variant) || !spec.getString("precision").equals("fp32")))
            throw new IllegalArgumentException("Custom candidates require a supported FP32 profile");
        String candidateFile=spec.getJSONObject("lengths").getJSONObject(Integer.toString(length)).getString("model");
        JSONObject hashes=assets.manifest.getJSONObject("files");
        JSONObject report=new JSONObject().put("schema_version",1).put("kind","model_optimization")
            .put("profile",name).put("candidate_variant",variant).put("trials",new JSONArray())
            .put("reference_sha256",hashes.get("minilm_vulkan_"+length+".pte"))
            .put("candidate_sha256",hashes.get(candidateFile)).put("candidate_precision",spec.getString("precision"))
            .put("config",spec.getJSONObject("config"));
        File output=output("optimization"); beginReport(output,report);
        try {
            report.put("layer_audit",auditModel(name,length,variant)); Assets.writeAtomic(output,report);
            if(!report.getJSONObject("layer_audit").getBoolean("correctness"))
                throw new IllegalStateException("Strict checkpoint correctness failed; candidate rejected");
            if(command.equals("model-screen")) {
                report.getJSONArray("trials").put(trial(0,variant,length,false,0,"vulkan",name,"candidate"));
            } else if(!command.equals("model-audit")) {
                for(int pair=0;pair<pairs;pair++) {
                    for(boolean candidate:pair%2==0?new boolean[]{false,true}:new boolean[]{true,false}) {
                        progress.accept("Model pair "+(pair+1)+"/"+pairs+": "+(candidate?name:"reference"));
                        report.getJSONArray("trials").put(trial(pair,candidate?variant:0,length,false,0,"vulkan",candidate?name:null,candidate?"candidate":"stock"));
                        Assets.writeAtomic(output,report);
                    }
                }
                if(command.equals("model-sustained")) {
                    JSONObject sustained=new JSONObject(report.toString());
                    sustained.remove("current_trial"); sustained.put("trials",new JSONArray());
                    report.put("sustained",sustained);
                    for(int pair=0;pair<5;pair++) {
                        for(boolean candidate:pair%2==0?new boolean[]{false,true}:new boolean[]{true,false}) {
                            progress.accept("Sustained model pair "+(pair+1)+"/5: "+(candidate?name:"reference"));
                            sustained.getJSONArray("trials").put(trial(pair,candidate?variant:0,length,false,sustainedSeconds/10.0,"vulkan",candidate?name:null,candidate?"candidate":"stock"));
                            Assets.writeAtomic(output,report);
                        }
                    }
                    sustained.put("status","complete");
                }
            }
            report.remove("current_trial"); report.put("status","complete");
            JSONObject acceptance=ModelAcceptance.evaluate(report, true);
            report.put("model_acceptance",acceptance);
            Assets.writeAtomic(output,report);
            if(acceptance.getBoolean("selection_eligible")) {
                JSONObject selection=metadata(length,false).put("profile",name).put("variant",variant)
                    .put("candidate_sha256",hashes.get(candidateFile)).put("report",output.getName())
                    .put("report_sha256",Assets.digest(output));
                Assets.writeAtomic(modelSelectionFile(length),selection);
            }
            return output;
        } catch(Exception e) {
            report.put("status","failed").put("error",e.toString()); Assets.writeAtomic(output,report); throw e;
        }
    }
    public File baseline(int length) throws Exception {
        File output = output("baseline");
        JSONObject report = new JSONObject().put("schema_version", 1).put("kind", "baseline")
            .put("trials", new JSONArray());
        beginReport(output, report);
        try {
            report.put("cpu_baseline", trial(-1, 0, length, false, 0, "cpu"));
            report.getJSONArray("trials").put(trial(0, 0, length, false, 0, "vulkan"));
            report.remove("current_trial"); report.put("status", "complete");
            Assets.writeAtomic(output, report); return output;
        } catch (Exception e) {
            report.put("status", "failed").put("error", e.toString());
            Assets.writeAtomic(output, report); throw e;
        }
    }
    public File paired(int length, int variant, boolean vr, boolean sustained) throws Exception {
        if (!Acceptance.allowsVariant(caps, variant)) throw new IllegalArgumentException("No such candidate in this device profile");
        JSONObject report = new JSONObject().put("schema_version", 1).put("trials", new JSONArray());
        File output = output(vr ? "vr" : "isolated");
        beginReport(output, report);
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
            report.remove("current_trial");
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
        File output = output("layers");
        JSONObject report = new JSONObject().put("schema_version", 1).put("kind", "layers")
            .put("capabilities", caps).put("precision", "fp32").put("model_revision", assets.manifest.get("model_revision"))
            .put("suite_revision", assets.manifest.get("suite_revision")).put("results", results);
        beginReport(output, report);
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
