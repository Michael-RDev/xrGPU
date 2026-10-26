package org.xrgpu;

import java.io.File;
import java.util.Objects;

/** Offline model inference with automatic qualified kernel selection. */
public final class Session implements AutoCloseable {
    static { System.loadLibrary("xrgpu_runtime"); }
    private long handle;
    private Session(long handle) { this.handle = handle; }
    public static Session load(File bundleDirectory) { return load(bundleDirectory, new Options()); }
    public static Session load(File bundleDirectory, Options options) {
        Objects.requireNonNull(bundleDirectory, "bundleDirectory");
        Objects.requireNonNull(options, "options");
        return new Session(load0(bundleDirectory.getAbsolutePath(), options.forceCpu, options.profiling,
                options.experiment, options.experimentalRepairs, options.repairsOnly, options.experimentalRules));
    }
    public synchronized Tensor[] forward(Tensor... inputs) {
        requireOpen();
        Objects.requireNonNull(inputs, "inputs");
        for (Tensor input : inputs) Objects.requireNonNull(input, "input");
        return forward0(handle, inputs);
    }
    /** JSON containing tensor specs, selected backend, decisions, and fallback reason. */
    public synchronized String info() { requireOpen(); return info0(handle); }
    @Override public synchronized void close() {
        if (handle != 0) { close0(handle); handle = 0; }
    }
    private void requireOpen() { if (handle == 0) throw new IllegalStateException("Closed session"); }
    public static final class Options {
        public boolean forceCpu = false;
        public boolean profiling = false;
        /** Development builds only; production rejects candidate overrides. */
        public String experiment = "";
        public boolean experimentalRepairs = false;
        public String experimentalRules = "[]";
        public boolean repairsOnly = false;
    }
    private static native long load0(String path, boolean forceCpu, boolean profiling,
            String experiment, boolean experimentalRepairs, boolean repairsOnly, String experimentalRules);
    private static native Tensor[] forward0(long handle, Tensor[] inputs);
    private static native String info0(long handle);
    private static native void close0(long handle);
}
