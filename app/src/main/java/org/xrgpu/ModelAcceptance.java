package org.xrgpu;

import org.json.*;
import java.util.*;

/** Separate program-comparison gate; the existing kernel-only gate is unchanged. */
public final class ModelAcceptance {
    static boolean commonIdentity(JSONObject a, JSONObject b) throws Exception {
        for(String key: Acceptance.IDENTITY) {
            if(key.equals("model_sha256") || key.equals("precision")) continue;
            if(!a.has(key) || a.isNull(key) || !b.has(key) || b.isNull(key) ||
                !a.get(key).toString().equals(b.get(key).toString())) return false;
        }
        return true;
    }
    static void require(boolean valid,String reason) {
        if(!valid) throw new IllegalArgumentException(reason);
    }
    static ArrayList<Double> times(JSONObject trial,String key) throws Exception {
        ArrayList<Double> values=new ArrayList<>(); JSONArray samples=trial.getJSONArray("samples");
        for(int i=0;i<samples.length();i++) {
            JSONObject sample=samples.getJSONObject(i); double value=sample.getDouble(key);
            require(Double.isFinite(value) && value>0,"invalid timing");
            require(sample.getInt("thermal_status")==trial.getInt("thermal_start") &&
                sample.getBoolean("charging")==trial.getBoolean("charging"),"thermal or charging transition");
            values.add(value);
        }
        return values;
    }
    public static JSONObject evaluate(JSONObject report,boolean sustainedRequired) throws Exception {
        try {
            require(report.getInt("schema_version")==1 && report.getString("kind").equals("model_optimization") &&
                report.getString("status").equals("complete"),"incomplete model report");
            JSONObject audit=report.getJSONObject("layer_audit");
            require(audit.getBoolean("correctness") && audit.getDouble("atol")==1e-4 && audit.getDouble("rtol")==1e-4 &&
                audit.getInt("checkpoints")==62,"strict checkpoint audit failed");
            JSONArray cases=audit.getJSONArray("cases"); require(cases.length()>=3,"insufficient audit cases");
            for(int c=0;c<cases.length();c++) {
                JSONArray checks=cases.getJSONObject(c).getJSONArray("checks"); require(checks.length()==62,"missing checkpoint");
                for(int j=0;j<checks.length();j++) {
                    JSONObject check=checks.getJSONObject(j); double ratio=check.getDouble("max_tolerance_ratio");
                    require(check.getBoolean("correctness") && Double.isFinite(ratio) && ratio>=0 && ratio<=1,"checkpoint tolerance exceeded");
                }
            }
            JSONArray trials=report.getJSONArray("trials");
            require(trials.length()>=8 && trials.length()%2==0,"four complete pairs required");
            JSONObject identity=trials.getJSONObject(0);
            ArrayList<Double> stock=new ArrayList<>(), candidate=new ArrayList<>(), stockModel=new ArrayList<>(), candidateModel=new ArrayList<>();
            double duration=0;
            for(int i=0;i<trials.length();i+=2) {
                JSONObject a=trials.getJSONObject(i), b=trials.getJSONObject(i+1);
                require(a.getInt("pair")==i/2 && b.getInt("pair")==i/2,"invalid pair order");
                String first=(i/2)%2==0?"stock":"candidate";
                require(a.getString("arm").equals(first) && !b.getString("arm").equals(first),"order did not alternate");
                require(a.getInt("thermal_start")==b.getInt("thermal_start"),"different starting thermal states");
                double am=0,bm=0;
                for(JSONObject t:new JSONObject[]{a,b}) {
                    boolean isCandidate=t.getString("arm").equals("candidate");
                    require(isCandidate || t.getString("arm").equals("stock"),"invalid arm");
                    int variant=isCandidate?report.getInt("candidate_variant"):0;
                    require(commonIdentity(t,identity) && t.getString("mode").equals("isolated") && t.getString("backend").equals("vulkan") &&
                        t.getInt("inference_interval_ms")==100,"incomparable workload or runtime");
                    String sha=report.getString(isCandidate?"candidate_sha256":"reference_sha256");
                    require(sha.matches("[0-9a-f]{64}") && t.getString("model_sha256").equals(sha) &&
                        t.getString("precision").equals(isCandidate?report.getString("candidate_precision"):"fp32") &&
                        t.getString("model_profile").equals(isCandidate?report.getString("profile"):"reference") &&
                        t.getInt("variant")==variant,"program identity mismatch");
                    require(variant==0?t.getLong("custom_nodes")==0:
                        Acceptance.allowsVariant(t,variant) && t.getLong("custom_nodes")>0,"dispatch mismatch");
                    require(t.getBoolean("correctness") && t.getString("status").equals("ok") &&
                        !t.getBoolean("thermal_discontinuity") && t.getInt("thermal_start")>=0 && t.getInt("thermal_start")<3 &&
                        t.getInt("warmup_iterations")>=20 && t.getJSONArray("samples").length()>=100,"invalid trial");
                    ArrayList<Double> values=times(t,"end_to_end_ms"), models=times(t,"model_ms");
                    if(isCandidate) { candidate.addAll(values); candidateModel.addAll(models); bm=Acceptance.percentile(values,.5); }
                    else { stock.addAll(values); stockModel.addAll(models); am=Acceptance.percentile(values,.5); }
                    double seconds=t.getDouble("measurement_seconds");
                    require(Double.isFinite(seconds) && seconds>0,"invalid duration"); duration+=seconds;
                }
                require(bm<am,"improvement did not repeat in every pair");
            }
            double reduction=1-Acceptance.percentile(candidate,.5)/Acceptance.percentile(stock,.5);
            require(reduction>=.1,"median reduction below 10%");
            require(Acceptance.percentile(candidate,.95)<=1.05*Acceptance.percentile(stock,.95) &&
                Acceptance.percentile(candidateModel,.95)<=1.05*Acceptance.percentile(stockModel,.95),"p95 regression");
            if(sustainedRequired) {
                JSONObject longer=report.getJSONObject("sustained");
                for(String key:new String[]{"profile","candidate_variant","candidate_sha256","reference_sha256","candidate_precision"})
                    require(report.get(key).equals(longer.get(key)),"sustained program mismatch");
                JSONArray longTrials=longer.getJSONArray("trials");
                for(int i=0;i<longTrials.length();i++) require(commonIdentity(identity,longTrials.getJSONObject(i)),"sustained runtime mismatch");
                JSONObject checked=evaluate(longer,false);
                require(checked.getBoolean("accepted") && checked.getDouble("measurement_seconds")>=600,"ten-minute comparison did not pass");
            }
            return new JSONObject().put("accepted",true).put("selection_eligible",sustainedRequired)
                .put("latency_reduction_percent",100*reduction).put("measurement_seconds",duration);
        } catch(Exception invalid) {
            return new JSONObject().put("accepted",false).put("selection_eligible",false).put("reason",invalid.toString());
        }
    }
}
