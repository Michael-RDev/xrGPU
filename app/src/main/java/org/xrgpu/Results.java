package org.xrgpu;

import org.json.*;
import java.util.*;

public final class Results {
    private static String number(double value) { return String.format(Locale.ROOT, "%.3f", value); }
    private static String distribution(JSONArray samples, String key) throws Exception {
        ArrayList<Double> values = new ArrayList<>();
        for (int i=0; i<samples.length(); i++) {
            JSONObject sample = samples.getJSONObject(i);
            if (sample.isNull(key)) return "unavailable";
            values.add(sample.getDouble(key));
        }
        if (values.isEmpty()) return "unavailable";
        return "median " + number(Acceptance.percentile(values, .5)) + " / p95 " +
            number(Acceptance.percentile(values, .95)) + " ms (n=" + values.size() + ")";
    }
    private static void profile(StringBuilder out, JSONObject profile) throws Exception {
        if (profile == null) return;
        JSONArray events = profile.optJSONArray("gpu_events");
        if (events == null || events.length() == 0) { out.append("GPU operation timings unavailable.\n"); return; }
        ArrayList<JSONObject> sorted = new ArrayList<>();
        for (int i=0; i<events.length(); i++) sorted.add(events.getJSONObject(i));
        sorted.sort((a,b) -> Double.compare(b.optDouble("ms"), a.optDouble("ms")));
        out.append("Slowest GPU operations (first measured inference):\n");
        for (int i=0; i<Math.min(8,sorted.size()); i++) {
            JSONObject event=sorted.get(i);
            out.append(number(event.getDouble("ms"))).append(" ms · ").append(event.getString("name")).append('\n');
        }
    }
    public static String summary(JSONObject report) throws Exception {
        StringBuilder out=new StringBuilder(report.optString("status", "incomplete")).append('\n');
        if (report.has("error")) out.append(report.getString("error")).append('\n');
        if (report.optString("kind").equals("layers")) {
            JSONArray results=report.getJSONArray("results");
            for(int i=0; i<results.length(); i++) {
                JSONObject r=results.getJSONObject(i);
                out.append(r.getString("case")).append(" · variant ").append(r.getInt("variant"))
                    .append("\nModel: ").append(distribution(r.getJSONArray("samples"),"model_ms"))
                    .append("\nGPU: ").append(distribution(r.getJSONArray("samples"),"gpu_ms")).append("\n\n");
            }
            return out.toString();
        }
        if (report.has("acceptance")) out.append(report.getJSONObject("acceptance").toString(2)).append("\n\n");
        if (report.has("cpu_baseline")) out.append("CPU reference: ")
            .append(distribution(report.getJSONObject("cpu_baseline").getJSONArray("samples"),"end_to_end_ms")).append('\n');
        JSONArray trials=report.optJSONArray("trials");
        if (trials == null) return out.toString();
        for (String arm : new String[]{"stock","candidate"}) {
            JSONArray samples=new JSONArray(); JSONObject first=null;
            for(int i=0; i<trials.length(); i++) {
                JSONObject trial=trials.getJSONObject(i);
                if(!trial.optString("arm").equals(arm)) continue;
                if(first==null) first=trial;
                JSONArray values=trial.getJSONArray("samples");
                for(int j=0;j<values.length();j++) samples.put(values.getJSONObject(j));
            }
            out.append('\n').append(arm).append("\nText → embedding: ").append(distribution(samples,"end_to_end_ms"))
                .append("\nModel: ").append(distribution(samples,"model_ms"))
                .append("\nGPU operations: ").append(distribution(samples,"gpu_ms")).append('\n');
            if(first!=null) {
                out.append("Cold load: ").append(number(first.getDouble("cold_load_ms"))).append(" ms\n");
                out.append("Custom dispatch nodes: ").append(first.getLong("custom_nodes")).append('\n');
                profile(out,first.optJSONObject("profile"));
            }
        }
        out.append("\nFull timing samples, thermal status, memory, and frame counters are in the saved local report.");
        return out.toString();
    }
}
