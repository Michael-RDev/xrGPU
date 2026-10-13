package org.xrgpu;

import android.app.NativeActivity;
import android.os.Bundle;
import android.view.WindowManager;
import android.util.Log;
import org.json.JSONObject;
import java.util.concurrent.atomic.AtomicBoolean;

public final class XrActivity extends NativeActivity {
    private final AtomicBoolean cancel = new AtomicBoolean();
    private Thread worker;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        worker = new Thread(() -> {
            try {
                Assets assets = new Assets(this);
                long deadline = System.currentTimeMillis() + 120000;
                while (!cancel.get()) {
                    JSONObject status = new JSONObject(Native.sceneStatus());
                    if (!status.optString("error").isEmpty()) throw new IllegalStateException(status.getString("error"));
                    if (status.optBoolean("ready") && status.optBoolean("focused") && Math.abs(status.optDouble("refresh_hz", 0) - 72) < .1) break;
                    if (System.currentTimeMillis() > deadline) throw new IllegalStateException("VR session did not become focused at 72 Hz");
                    Thread.sleep(100);
                }
                if (!cancel.get()) new BenchmarkRunner(this, assets, cancel, s -> Log.i("xrGPU", s))
                    .paired(getIntent().getIntExtra("length",128), getIntent().getIntExtra("variant",1), true,
                        getIntent().getBooleanExtra("sustained",true));
            } catch (Exception | LinkageError e) {
                Log.e("xrGPU", "VR benchmark failed", e);
                try { Assets.writeAtomic(new java.io.File(getFilesDir(), "results/" + System.currentTimeMillis() + "-vr-error.json"),
                    new JSONObject().put("schema_version",1).put("status","failed").put("error",e.toString())); } catch(Exception ignored) { Log.e("xrGPU", "Could not save error report", ignored); }
            } finally { runOnUiThread(this::finish); }
        }, "xrGPU-inference"); worker.start();
    }
    @Override protected void onPause() { cancel.set(true); super.onPause(); }
    @Override protected void onDestroy() { cancel.set(true); if(worker!=null) worker.interrupt(); super.onDestroy(); }
}
