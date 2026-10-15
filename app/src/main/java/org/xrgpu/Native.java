package org.xrgpu;

public final class Native {
    static { System.loadLibrary("xrgpu"); }
    public static native String capabilities();
    public static native String sceneStats(boolean reset);
    public static native String sceneStatus();
    private static native long create(String path, int variant);
    private static native void destroy(long handle);
    private static native float[] tokens(long handle, long[] input, long[] mask);
    private static native float[] layer(long handle, float[] input, int rows, int columns);
    private static native String stats(long handle);
    private static native boolean supportsLinear(long handle, int rows, int columns, int outputs);
    public static final class Model implements AutoCloseable {
        private long handle;
        public Model(String path, int variant) { handle = create(path, variant); }
        public float[] encode(WordPiece.Tokens t) { return tokens(handle, t.ids, t.mask); }
        public float[] layer(float[] input, int rows, int columns) { return Native.layer(handle, input, rows, columns); }
        public String stats() { return Native.stats(handle); }
        public boolean supportsLinear(int rows, int columns, int outputs) { return Native.supportsLinear(handle, rows, columns, outputs); }
        @Override public void close() { if (handle != 0) { destroy(handle); handle = 0; } }
    }
}
