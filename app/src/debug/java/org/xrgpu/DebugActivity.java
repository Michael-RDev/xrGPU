package org.xrgpu;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.TextView;
import org.json.JSONObject;
import java.io.File;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/** Present only in debug APKs. No changes to the production acceptance gate. */
public final class DebugActivity extends Activity {
    private final AtomicBoolean cancel = new AtomicBoolean();
    private volatile boolean busy;
    private String currentId;
    private TextView view;
    private Thread worker;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        view = new TextView(this); view.setTextSize(20); view.setPadding(24,24,24,24);
        setContentView(view); dispatch(getIntent());
    }
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent); dispatch(intent);
    }
    private void dispatch(Intent intent) {
        String id = intent.getStringExtra("run_id");
        String command = intent.getStringExtra("command");
        if (id == null || !id.matches("[a-zA-Z0-9_-]{1,80}")) {
            view.setText("Missing or invalid run_id"); return;
        }
        if ("cancel".equals(command)) {
            if (id.equals(currentId)) cancel.set(true);
            return;
        }
        File statusFile = new File(getFilesDir(), "commands/" + id + ".json");
        // Never overwrite evidence or start the same command twice.
        if (statusFile.exists()) { view.setText("Run ID already exists: " + id); return; }
        JSONObject status = new JSONObject();
        try {
            status.put("schema_version",1).put("run_id",id).put("command",command)
                .put("report", "results/" + id + ".json");
            int length = intent.getIntExtra("length",128), variant = intent.getIntExtra("variant",0);
            if (busy) throw new IllegalStateException("Another command is running");
            if (!Arrays.asList("inspect","correctness","baseline","layers","paired",
                    "model-audit","model-screen","model-paired","model-sustained").contains(command))
                throw new IllegalArgumentException("Unknown command");
            if (length != 32 && length != 128 && length != 256)
                throw new IllegalArgumentException("Unsupported sequence length");
            status.put("status","running"); Assets.writeAtomic(statusFile,status);
            busy=true; currentId=id; cancel.set(false);
            worker = new Thread(() -> {
                try {
                    File report = new File(getFilesDir(),"results/" + id + ".json");
                    if (command.equals("inspect")) {
                        Assets.writeAtomic(report,new JSONObject(Native.capabilities()).put("status","complete")
                            .put("thermal_status",getSystemService(android.os.PowerManager.class).getCurrentThermalStatus())
                            .put("charging",getSystemService(android.os.BatteryManager.class).isCharging()));
                    } else {
                        Assets assets = new Assets(this);
                        BenchmarkRunner runner = new BenchmarkRunner(this,assets,cancel,s -> {
                            android.util.Log.i("xrGPU",id + ": " + s);
                            runOnUiThread(() -> view.setText(id + "\n" + s));
                        });
                        runner.commandOutput=report;
                        switch(command) {
                            case "model-audit": case "model-screen": case "model-paired": case "model-sustained":
                                runner.modelExperiment(command,intent.getStringExtra("profile"),length,variant); break;
                            case "baseline": runner.baseline(length); break;
                            case "layers": runner.layers(variant); break;
                            case "paired": runner.paired(length,variant,false,false); break;
                            case "correctness":
                                runner.check(false);
                                try (Native.Model model = new Native.Model(assets.model("vulkan",length),variant)) {
                                    Validation.embeddings(assets,model,length);
                                }
                                Assets.writeAtomic(report,runner.metadata(length,false)
                                    .put("status","complete").put("correctness",true).put("variant",variant));
                                break;
                            default: throw new IllegalArgumentException("Unknown command");
                        }
                    }
                    status.put("status","complete");
                } catch (Exception | LinkageError e) {
                    try { status.put("status",cancel.get()?"canceled":"failed").put("error",e.toString()); }
                    catch (Exception ignored) { android.util.Log.e("xrGPU","Status error",ignored); }
                    android.util.Log.e("xrGPU",id + " failed",e);
                } finally {
                    try { Assets.writeAtomic(statusFile,status); }
                    catch (Exception e) { android.util.Log.e("xrGPU","Cannot save command status",e); }
                    busy=false;
                    runOnUiThread(() -> view.setText(status.toString()));
                }
            },"xrGPU-debug");
            worker.start();
        } catch (Exception e) {
            try { status.put("status","failed").put("error",e.toString()); Assets.writeAtomic(statusFile,status); }
            catch (Exception nested) { android.util.Log.e("xrGPU","Cannot reject command",nested); }
        }
    }
    @Override protected void onStop() { cancel.set(true); super.onStop(); }
    @Override protected void onDestroy() {
        cancel.set(true); if(worker!=null) worker.interrupt(); super.onDestroy();
    }
}
