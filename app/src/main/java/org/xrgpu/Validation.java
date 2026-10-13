package org.xrgpu;
import org.json.*;
import java.util.*;

public final class Validation {
    public static double cosine(float[] a, float[] b) {
        if (a.length != b.length || a.length == 0) return -1;
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.length; i++) {
            if (!Float.isFinite(a[i]) || !Float.isFinite(b[i])) return -1;
            dot += a[i] * (double)b[i]; aa += a[i] * (double)a[i]; bb += b[i] * (double)b[i];
        }
        return aa > 0 && bb > 0 ? dot / Math.sqrt(aa * bb) : -1;
    }
    public static float[] floats(JSONArray values) throws JSONException {
        float[] result = new float[values.length()];
        for (int i = 0; i < result.length; i++) result[i] = (float)values.getDouble(i);
        return result;
    }
    public static void embeddings(Assets assets, Native.Model model, int length) throws Exception {
        JSONArray references = new JSONArray(assets.text("references.json"));
        Map<String, float[]> outputs = new HashMap<>();
        for (int i = 0; i < references.length(); i++) {
            JSONObject reference = references.getJSONObject(i);
            if (reference.getInt("sequence_length") != length) continue;
            String text = reference.getString("text");
            WordPiece.Tokens tokens = assets.tokenizer.encode(text, length);
            JSONArray expectedIds = reference.getJSONArray("input_ids"), expectedMask = reference.getJSONArray("attention_mask");
            for (int j = 0; j < length; j++) {
                if (tokens.ids[j] != expectedIds.getLong(j) || tokens.mask[j] != expectedMask.getLong(j))
                    throw new IllegalStateException("Tokenizer mismatch at token " + j + " for " + text);
            }
            float[] output = model.encode(tokens);
            double normSquared=0; for(float value:output) normSquared+=value*(double)value;
            if (output.length != 384 || !Double.isFinite(normSquared) || Math.abs(Math.sqrt(normSquared)-1) > .001 ||
                cosine(output, floats(reference.getJSONArray("embedding"))) < .999)
                throw new IllegalStateException("Embedding correctness failed for " + text);
            outputs.put(text, output);
        }
        JSONObject corpus = new JSONObject(assets.text("corpus.json"));
        JSONArray docs = corpus.getJSONArray("documents"), queries = corpus.getJSONArray("queries");
        for (int q = 0; q < queries.length(); q++) {
            JSONObject query = queries.getJSONObject(q); double best = -2; String winner = "";
            for (int d = 0; d < docs.length(); d++) {
                JSONObject doc = docs.getJSONObject(d);
                double score = cosine(outputs.get(query.getString("text")), outputs.get(doc.getString("text")));
                if (score > best) { best = score; winner = doc.getString("id"); }
            }
            if (!winner.equals(query.getString("expected_id"))) throw new IllegalStateException("Retrieval mismatch: " + query);
        }
    }
    public static void layer(float[] actual, float[] expected) {
        if (actual.length != expected.length) throw new IllegalStateException("Layer output shape mismatch");
        for (int i = 0; i < actual.length; i++)
            if (!Float.isFinite(actual[i]) || Math.abs(actual[i] - expected[i]) > 1e-4 + 1e-4 * Math.abs(expected[i]))
                throw new IllegalStateException("Layer mismatch at " + i);
    }
}
