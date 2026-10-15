package org.xrgpu;
import org.json.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;

public final class HostTests {
    static void require(boolean value, String message) { if (!value) throw new AssertionError(message); }
    public static void main(String[] args) throws Exception {
        Path bundle = Path.of(args[0]);
        WordPiece tokenizer = new WordPiece(Files.newInputStream(bundle.resolve("tokenizer/vocab.txt")));
        JSONArray references = new JSONArray(Files.readString(bundle.resolve("references.json")));
        for (int i=0; i<references.length(); i++) {
            JSONObject c=references.getJSONObject(i); WordPiece.Tokens t=tokenizer.encode(c.getString("text"),c.getInt("sequence_length"));
            for(int j=0;j<t.ids.length;j++) {
                require(t.ids[j]==c.getJSONArray("input_ids").getLong(j),"Tokenizer mismatch case="+i+" token="+j);
                require(t.mask[j]==c.getJSONArray("attention_mask").getLong(j),"Mask mismatch");
            }
        }
        JSONObject evidence=new JSONObject(Files.readString(Path.of(args[1])));
        require(Acceptance.evaluate(evidence,true).getBoolean("accepted"),"Valid evidence rejected");
        evidence.getJSONArray("trials").getJSONObject(1).put("custom_nodes",0);
        require(!Acceptance.evaluate(evidence,true).getBoolean("accepted"),"Undispatched candidate accepted");
        evidence=new JSONObject(Files.readString(Path.of(args[1])));
        JSONObject candidate=evidence.getJSONArray("trials").getJSONObject(1);
        JSONObject changed=new JSONObject(candidate.toString()).put("device_profile","another-headset");
        require(!Acceptance.matchesIdentity(candidate,changed),"Cross-profile selection accepted");
        changed=new JSONObject(candidate.toString()).put("kernel_set","another-kernel-set");
        require(!Acceptance.matchesIdentity(candidate,changed),"Cross-kernel-set selection accepted");
        candidate.put("device_profile","another-headset");
        require(!Acceptance.evaluate(evidence,true).getBoolean("accepted"),"Mixed profiles accepted");
        evidence=new JSONObject(Files.readString(Path.of(args[1])));
        for (JSONObject run : new JSONObject[]{evidence,evidence.getJSONObject("sustained")}) {
            JSONArray trials=run.getJSONArray("trials");
            for(int i=0;i<trials.length();i++) {
                JSONObject t=trials.getJSONObject(i);
                t.put("device_profile","test-only").put("kernel_set","test-only.linear")
                    .put("candidate_variants",new JSONArray().put(7).put(19));
                if(t.getString("arm").equals("candidate")) t.put("variant",7);
            }
        }
        require(Acceptance.evaluate(evidence,true).getBoolean("accepted"),"Profile-specific variant IDs rejected");
        evidence.getJSONArray("trials").getJSONObject(1).remove("device_profile");
        require(!Acceptance.evaluate(evidence,true).getBoolean("accepted"),"Legacy evidence accepted without profile");
        System.out.println("PASS: "+references.length()+" Hugging Face tokenizer cases; acceptance gate checks");
    }
}
