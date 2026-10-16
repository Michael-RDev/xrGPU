package org.xrgpu;
import org.json.*;
import java.util.*;

public final class Acceptance {
    public static final String[] IDENTITY = {"device", "device_profile", "kernel_set", "candidate_variants",
        "gpu", "vendor_id", "device_id", "api_version", "driver", "os", "model_revision", "model_sha256",
        "runtime_revision", "kernel_revision", "precision", "sequence_length", "mode", "scene_revision",
        "refresh_hz", "inference_interval_ms", "suite_revision", "charging"};
    public static boolean matchesIdentity(JSONObject a, JSONObject b) {
        for (String key : IDENTITY)
            if (!a.has(key) || a.isNull(key) || !b.has(key) || b.isNull(key) ||
                !a.opt(key).toString().equals(b.opt(key).toString())) return false;
        return true;
    }
    public static boolean allowsVariant(JSONObject metadata, int variant) {
        JSONArray variants = metadata.optJSONArray("candidate_variants");
        if (variant <= 0 || variants == null || metadata.optString("device_profile", "stock").equals("stock")) return false;
        for (int i=0; i<variants.length(); i++) if (variants.optInt(i, 0) == variant) return true;
        return false;
    }
    public static double percentile(List<Double> input, double q) {
        if (input.isEmpty()) throw new IllegalArgumentException("Missing measurements");
        ArrayList<Double> values = new ArrayList<>(input); Collections.sort(values);
        for (double value : values) if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("Invalid timing");
        double p = (values.size() - 1) * q; int a = (int)Math.floor(p), b = (int)Math.ceil(p);
        return values.get(a) + (values.get(b) - values.get(a)) * (p - a);
    }
    private static ArrayList<Double> timings(JSONArray samples) throws JSONException {
        ArrayList<Double> out = new ArrayList<>();
        for (int i = 0; i < samples.length(); i++) out.add(samples.getJSONObject(i).getDouble("end_to_end_ms"));
        return out;
    }
    private static ArrayList<Double> numbers(JSONArray values) throws JSONException {
        ArrayList<Double> out = new ArrayList<>();
        for (int i = 0; i < values.length(); i++) out.add(values.getDouble(i));
        return out;
    }
    public static JSONObject evaluate(JSONObject report, boolean sustainedRequired) throws Exception {
        TreeSet<String> reasons = new TreeSet<>();
        if(report.optInt("schema_version")!=1) throw new IllegalArgumentException("Unsupported report schema");
        if (report.optString("status").equals("failed")) reasons.add("run failed or was interrupted");
        JSONArray trials = report.getJSONArray("trials");
        Map<Integer, Map<String, JSONObject>> pairs = new TreeMap<>();
        JSONObject identity = trials.length() > 0 ? trials.getJSONObject(0) : new JSONObject();
        int candidateVariant = -1;
        for (int i = 0; i < trials.length(); i++) {
            JSONObject t = trials.getJSONObject(i);
            if (!matchesIdentity(t, identity)) reasons.add("incomparable metadata");
            if (t.getString("arm").equals("candidate")) {
                if (candidateVariant < 0) candidateVariant = t.getInt("variant");
                if (!allowsVariant(t, candidateVariant) || candidateVariant != t.getInt("variant")) reasons.add("invalid candidate attribution");
            } else if (!t.getString("arm").equals("stock") || t.getInt("variant") != 0) reasons.add("invalid stock attribution");
            Map<String, JSONObject> pair = pairs.computeIfAbsent(t.getInt("pair"), k -> new HashMap<>());
            if (pair.put(t.getString("arm"), t) != null) reasons.add("duplicate trial arm");
        }
        if (pairs.size() < 4) reasons.add("requires four paired trials");
        ArrayList<Double> allA = new ArrayList<>(), allB = new ArrayList<>();
        for (Map<String, JSONObject> pair : pairs.values()) {
            if (!pair.containsKey("stock") || !pair.containsKey("candidate")) { reasons.add("incomplete pair"); continue; }
            JSONObject a = pair.get("stock"), b = pair.get("candidate");
            if(a.optInt("thermal_start",-1)<0 || a.optInt("thermal_start",-1)!=b.optInt("thermal_start",-2)) reasons.add("different or unavailable starting thermal state");
            for (JSONObject t : new JSONObject[]{a, b}) {
                if (!t.optBoolean("correctness") || !t.optString("status").equals("ok")) reasons.add("correctness did not pass");
                if (t.optBoolean("thermal_discontinuity", true)) reasons.add("thermal state changed or was unavailable");
                if (t.optInt("warmup_iterations") < 20 || t.getJSONArray("samples").length() < 100) reasons.add("insufficient samples");
            }
            if (b.optLong("custom_nodes") <= 0) reasons.add("candidate was not dispatched");
            ArrayList<Double> av = timings(a.getJSONArray("samples")), bv = timings(b.getJSONArray("samples"));
            if (av.isEmpty() || bv.isEmpty()) { reasons.add("missing timings"); continue; }
            if (percentile(bv, .5) <= 0 || percentile(av, .5) <= percentile(bv, .5)) reasons.add("speedup did not repeat in every trial");
            allA.addAll(av); allB.addAll(bv);
            if (a.getString("mode").equals("vr")) {
                JSONArray af = a.optJSONArray("frame_ms"), bf = b.optJSONArray("frame_ms");
                if (af == null || bf == null || af.length() == 0 || bf.length() == 0 ||
                    a.isNull("missed_frame_rate") || b.isNull("missed_frame_rate")) reasons.add("missing VR metrics");
                else {
                    for (JSONObject t : new JSONObject[]{a,b}) {
                        double missed=t.getDouble("missed_frame_rate");
                        if (!Double.isFinite(missed) || missed<0 || missed>1) reasons.add("invalid missed-frame metric");
                    }
                    if (percentile(numbers(bf), .95) > percentile(numbers(af), .95) * 1.05) reasons.add("VR p95 regression exceeds 5%");
                    if (b.getDouble("missed_frame_rate") - a.getDouble("missed_frame_rate") > .01) reasons.add("missed-frame regression exceeds one percentage point");
                }
                if (Math.abs(a.optDouble("refresh_hz", 0) - 72) > .1 || Math.abs(b.optDouble("refresh_hz", 0) - 72) > .1)
                    reasons.add("72 Hz was not maintained");
            }
        }
        double speedup = !allA.isEmpty() && !allB.isEmpty() && percentile(allB, .5) > 0 ? percentile(allA, .5) / percentile(allB, .5) : 0;
        if (speedup < 1 / .9) reasons.add("median latency reduction is below 10%");
        if (!allA.isEmpty() && !allB.isEmpty() && percentile(allB, .95) > percentile(allA, .95) * 1.05) reasons.add("end-to-end p95 regression exceeds 5%");
        if (sustainedRequired) {
            JSONObject sustained = report.optJSONObject("sustained");
            double duration = 0; boolean matches = sustained != null;
            if (sustained != null) {
                JSONArray longTrials = sustained.getJSONArray("trials");
                for (int i=0; i<longTrials.length(); i++) {
                    JSONObject t=longTrials.getJSONObject(i); duration += t.optDouble("measurement_seconds",0);
                    matches &= matchesIdentity(t, identity);
                    if (t.optString("arm").equals("candidate")) matches &= t.optInt("variant") == candidateVariant;
                }
            }
            if (!matches || !Double.isFinite(duration) || duration < 600 || !evaluate(sustained, false).getBoolean("accepted"))
                reasons.add("ten-minute sustained comparison has not passed");
        }
        return new JSONObject().put("accepted", reasons.isEmpty()).put("median_speedup", speedup)
            .put("reasons", new JSONArray(reasons));
    }
}
