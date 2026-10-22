package org.xrgpu;

import android.os.*;
import org.json.*;
import java.io.*;
import java.nio.file.Files;
import java.util.UUID;

/** Portable stock-kernel survey; never enables a device-specific candidate. */
public final class LayerSuite {
    static JSONObject compare(float[] actual, float[] expected, double atol, double rtol) throws Exception {
        if (actual.length != expected.length) return new JSONObject().put("passed",false).put("reason","output size mismatch");
        double max=0, ratio=0;
        for(int i=0; i<actual.length; i++) {
            if(!Float.isFinite(actual[i]) || !Float.isFinite(expected[i]))
                return new JSONObject().put("passed",false).put("reason","non-finite output or reference");
            double error=Math.abs((double)actual[i]-expected[i]);
            max=Math.max(max,error); ratio=Math.max(ratio,error/(atol+rtol*Math.abs(expected[i])));
        }
        return new JSONObject().put("passed",ratio<=1).put("max_abs_error",max)
            .put("max_tolerance_ratio",ratio).put("atol",atol).put("rtol",rtol);
    }
    private static boolean sameShape(JSONArray actual, JSONArray expected) throws Exception {
        if(actual.length()!=expected.length()) return false;
        for(int i=0;i<actual.length();i++) if(actual.getInt(i)!=expected.getInt(i)) return false;
        return true;
    }
    private static void validate(Native.Model model,float[] actual,float[] expected,JSONObject test,
                                 JSONObject result,double atol,double rtol) throws Exception {
        JSONObject check=compare(actual,expected,atol,rtol);
        if(!sameShape(new JSONObject(model.stats()).getJSONArray("output_shape"),test.getJSONArray("output_shape")))
            check.put("passed",false).put("reason","output shape mismatch");
        result.put("correctness",check);
        if(!check.getBoolean("passed")) throw new IllegalStateException("Layer correctness failed");
    }
    public static File run(BenchmarkRunner runner, File output) throws Exception {
        Assets assets=runner.assets;
        if(!assets.manifest.getJSONObject("files").has("layer_suite/suite.json"))
            throw new IllegalStateException("Layer suite is not packaged. Run models.export_layers and rebuild the APK.");
        JSONObject suite=new JSONObject(assets.text("layer_suite/suite.json"));
        int warmups=suite.getInt("warmup_iterations"), iterations=suite.getInt("measured_iterations");
        double atol=suite.getDouble("atol"), rtol=suite.getDouble("rtol");
        JSONObject device=new JSONObject(runner.caps.toString()).put("manufacturer",Build.MANUFACTURER)
            .put("model",Build.MODEL).put("os",Build.FINGERPRINT);
        JSONArray results=new JSONArray(), cases=suite.getJSONArray("cases");
        JSONObject report=new JSONObject().put("schema_version",1).put("kind","layer_suite")
            .put("run_id",runner.commandOutput!=null?output.getName().replace(".json",""):"layers-"+UUID.randomUUID())
            .put("created_at",java.time.Instant.now().toString()).put("status","running")
            .put("execution_location","headset").put("backend","executorch_vulkan").put("variant",0)
            .put("device",device).put("suite_revision",suite.getString("suite_revision"))
            .put("suite_sha256",assets.manifest.getJSONObject("files").get("layer_suite/suite.json"))
            .put("export_environment",suite.getJSONObject("environment")).put("seed",suite.get("seed"))
            .put("precision","fp32").put("reference",suite.get("reference"))
            .put("warmup_iterations",warmups).put("measured_iterations",iterations).put("results",results)
            .put("timing_scope","execution_ms: JNI input copies + forward + output readback; model_ms: native forward; gpu_ms: sum of timestamped operations, not critical-path latency");
        Assets.writeAtomic(output,report);
        try {
            for(int i=0;i<cases.length();i++) {
                runner.check(false);
                JSONObject test=cases.getJSONObject(i);
                JSONObject result=new JSONObject().put("case",test.getString("id")).put("operation",test.get("operation"))
                    .put("inputs",test.get("inputs")).put("params",test.get("params")).put("output_shape",test.get("output_shape"))
                    .put("work",test.get("work")).put("status","failed").put("samples",new JSONArray());
                results.put(result);
                if(!test.getString("export_status").equals("ready") || !runner.caps.optBoolean("vulkan")) {
                    result.put("status","skipped").put("reason",test.optString("reason","Vulkan unavailable"));
                    Assets.writeAtomic(output,report); continue;
                }
                runner.progress.accept("Layer "+(i+1)+"/"+cases.length()+": "+test.getString("id"));
                try {
                    JSONArray specs=test.getJSONArray("inputs");
                    byte[][] inputs=new byte[specs.length()][];
                    int[][] shapes=new int[specs.length()][];
                    String[] dtypes=new String[specs.length()];
                    for(int j=0;j<specs.length();j++) {
                        JSONObject spec=specs.getJSONObject(j); JSONArray shape=spec.getJSONArray("shape");
                        inputs[j]=Files.readAllBytes(new File(assets.root,spec.getString("file")).toPath());
                        shapes[j]=new int[shape.length()];
                        for(int k=0;k<shape.length();k++) shapes[j][k]=shape.getInt(k);
                        dtypes[j]=spec.getString("dtype");
                    }
                    float[] expected=assets.floats(test.getString("expected"));
                    result.put("program_sha256",test.getString("program_sha256")).put("coverage",test.get("coverage"));
                    try(Native.Model model=new Native.Model(new File(assets.root,test.getString("program")).getPath(),0)) {
                        validate(model,model.tensors(inputs,shapes,dtypes),expected,test,result,atol,rtol);
                        for(int w=0;w<warmups;w++) { runner.check(false); model.tensors(inputs,shapes,dtypes); }
                        int initial=runner.power.getCurrentThermalStatus();
                        boolean charging=runner.context.getSystemService(BatteryManager.class).isCharging(), changed=false;
                        result.put("thermal_start",initial).put("charging_start",charging);
                        for(int n=0;n<iterations;n++) {
                            runner.check(false);
                            long start=SystemClock.elapsedRealtimeNanos();
                            float[] actual=model.tensors(inputs,shapes,dtypes);
                            double elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1e6;
                            JSONObject stats=new JSONObject(model.stats());
                            int thermal=runner.power.getCurrentThermalStatus();
                            boolean currentCharging=runner.context.getSystemService(BatteryManager.class).isCharging();
                            changed |= thermal!=initial || currentCharging!=charging;
                            result.getJSONArray("samples").put(new JSONObject().put("execution_ms",elapsed)
                                .put("model_ms",stats.get("model_ms")).put("gpu_ms",stats.get("gpu_ms"))
                                .put("input_staging_ms",stats.get("input_staging_ms")).put("output_staging_ms",stats.get("output_staging_ms"))
                                .put("submit_and_wait_ms",stats.get("submit_and_wait_ms"))
                                .put("thermal_status",thermal).put("charging",currentCharging).put("process_pss_kb",Debug.getPss()));
                            if(n==0) result.put("profile",stats).put("cold_load_ms",stats.get("load_ms"));
                            validate(model,actual,expected,test,result,atol,rtol);
                            result.put("thermal_discontinuity",changed);
                            if((n+1)%25==0) Assets.writeAtomic(output,report);
                        }
                        result.put("status","passed");
                    }
                } catch(InterruptedException e) { throw e; }
                catch(Exception e) { result.put("status","failed").put("error",e.toString()); }
                Assets.writeAtomic(output,report);
            }
            report.put("status","complete");
        } catch(Exception e) {
            report.put("status",e instanceof InterruptedException?"canceled":"failed").put("error",e.toString()); throw e;
        } finally { Assets.writeAtomic(output,report); }
        return output;
    }
}
