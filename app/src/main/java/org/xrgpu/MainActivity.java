package org.xrgpu;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancel = new AtomicBoolean();
    private volatile boolean busy = false;
    private Assets assets;
    private TextView status;
    private EditText query;
    private Spinner length, variant;
    private CheckBox sustained;
    private interface Work { void run() throws Exception; }
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ScrollView scroll = new ScrollView(this); LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL); panel.setPadding(32, 24, 32, 24); scroll.addView(panel); setContentView(scroll);
        TextView heading = new TextView(this); heading.setText("xrGPU\nModel layers on your headset"); heading.setTextSize(26); panel.addView(heading);
        TextView description = new TextView(this);
        description.setText("Offline MiniLM search and experiments. Candidate kernels are experimental until this device passes correctness, paired, and sustained checks."); panel.addView(description);
        length = spinner(panel, new String[]{"32 tokens", "128 tokens", "256 tokens"}); length.setSelection(1);
        variant = spinner(panel, new String[]{"Candidate 1: 4×8 tiles", "Candidate 2: 2×8 tiles", "Candidate 3: buffer weights"});
        sustained = new CheckBox(this); sustained.setText("Include ten-minute sustained comparison (required to enable a candidate)"); sustained.setChecked(true); panel.addView(sustained);
        addButton(panel, "Inspect headset", () -> submit(() -> show(new JSONObject(Native.capabilities()).toString(2))));
        addButton(panel, "Check model correctness", () -> { int n = selectedLength(); submit(() -> {
            try (Native.Model model = new Native.Model(assets.model("vulkan", n), 0)) { Validation.embeddings(assets, model, n); }
            show("Tokenizer, embeddings, and labeled retrieval passed at " + n + " tokens.");
        }); });
        addButton(panel, "Benchmark layer candidates", () -> { int v = variant.getSelectedItemPosition() + 1;
            submit(() -> { File report = runner().layers(v); show(report.getName() + "\n" + Results.summary(read(report))); }); });
        addButton(panel, "Benchmark model in isolation", () -> { int n = selectedLength(), v = variant.getSelectedItemPosition() + 1; boolean longRun = sustained.isChecked();
            submit(() -> { File report = runner().paired(n, v, false, longRun); show(report.getName() + "\n" + Results.summary(read(report))); }); });
        addButton(panel, "Benchmark alongside VR scene", () -> {
            if (busy) { show("Finish or cancel the current run first."); return; }
            startActivity(new Intent(this, XrActivity.class).putExtra("length", selectedLength())
                .putExtra("variant", variant.getSelectedItemPosition() + 1).putExtra("sustained", sustained.isChecked()));
        });
        addButton(panel, "Cancel current run", () -> { cancel.set(true); show("Stopping after the current inference..."); });
        query = new EditText(this); query.setHint("Search, e.g. ‘I need a light’"); query.setSingleLine(); panel.addView(query);
        addButton(panel, "Search local descriptions", () -> { String text = query.getText().toString(); int n = selectedLength(); submit(() -> search(text, n)); });
        addButton(panel, "View latest result", () -> submit(this::latest));
        status = new TextView(this); status.setTextIsSelectable(true); status.setTextSize(15); panel.addView(status);
        submit(() -> { assets = new Assets(this); show("Offline assets verified. Ready.\n" + new JSONObject(Native.capabilities()).toString(2)); });
    }
    private Spinner spinner(LinearLayout panel, String[] items) {
        Spinner s = new Spinner(this); s.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items)); panel.addView(s); return s;
    }
    private void addButton(LinearLayout panel, String label, Runnable action) {
        Button button = new Button(this); button.setText(label); button.setOnClickListener(v -> action.run()); panel.addView(button);
    }
    private int selectedLength() { return new int[]{32,128,256}[length.getSelectedItemPosition()]; }
    private void show(String text) { runOnUiThread(() -> status.setText(text.length() > 16000 ? text.substring(0,16000) + "\nFull data is in the saved report." : text)); }
    private void submit(Work action) {
        if (busy) { show("A run is already active."); return; }
        busy = true; cancel.set(false);
        worker.submit(() -> {
            try { action.run(); }
            catch (Exception | LinkageError e) { show("Run failed: " + e); }
            finally { busy = false; }
        });
    }
    private BenchmarkRunner runner() throws Exception {
        if (assets == null) throw new IllegalStateException("Assets are not ready");
        return new BenchmarkRunner(this, assets, cancel, this::show);
    }
    private int selectedVariant(int length) throws Exception {
        try {
        File selection = new File(getFilesDir(), "selections/" + length + "-isolated.json");
        if (!selection.exists()) return 0;
        JSONObject candidate = read(selection), current = runner().metadata(length, false);
        for (String key : Acceptance.IDENTITY) if (!current.get(key).equals(candidate.opt(key))) return 0;
        File report = new File(getFilesDir(), "results/" + candidate.getString("report"));
        if (!report.exists() || !Assets.digest(report).equals(candidate.getString("report_sha256"))) return 0;
        JSONObject evidence = read(report);
        if (!Acceptance.evaluate(evidence, true).getBoolean("accepted")) return 0;
        JSONArray trials = evidence.getJSONArray("trials");
        for (int i=0; i<trials.length(); i++) {
            JSONObject trial=trials.getJSONObject(i);
            if (trial.getString("arm").equals("candidate"))
                return trial.getInt("variant")==candidate.getInt("variant") ? candidate.getInt("variant") : 0;
        }
        return 0;
        } catch (Exception invalidEvidence) { return 0; }
    }
    private void search(String text, int length) throws Exception {
        int v = selectedVariant(length);
        try (Native.Model model = new Native.Model(assets.model("vulkan", length), v)) {
            float[] needle = model.encode(assets.tokenizer.encode(text, length));
            JSONArray docs = new JSONObject(assets.text("corpus.json")).getJSONArray("documents");
            double best = -2; String match = "";
            for (int i = 0; i < docs.length(); i++) {
                if (cancel.get()) throw new InterruptedException("Search canceled");
                JSONObject d = docs.getJSONObject(i);
                double score = Validation.cosine(needle, model.encode(assets.tokenizer.encode(d.getString("text"), length)));
                if (score > best) { best = score; match = d.getString("text"); }
            }
            show(match + "\nSimilarity: " + String.format(Locale.ROOT, "%.3f", best) + "\n" +
                (v == 0 ? "Stock Vulkan" : "Validated local candidate " + v) + " · input limit " + length + " tokens");
        }
    }
    private static JSONObject read(File f) throws Exception { return new JSONObject(new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8)); }
    private void latest() throws Exception {
        File[] reports = new File(getFilesDir(), "results").listFiles((dir,name) -> name.endsWith(".json"));
        if (reports == null || reports.length == 0) { show("No saved results yet."); return; }
        Arrays.sort(reports, Comparator.comparing(File::getName));
        File latest = reports[reports.length-1]; JSONObject report = read(latest);
        show(latest.getName() + "\n" + Results.summary(report));
    }
    @Override protected void onStop() { cancel.set(true); super.onStop(); }
    @Override protected void onDestroy() { cancel.set(true); worker.shutdownNow(); super.onDestroy(); }
}
