package org.xrgpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/** An immutable, owned FP32 or INT64 tensor with a static shape. */
public final class Tensor {
    public enum DType { FLOAT32, INT64 }
    private final String name;
    private final long[] shape;
    private final int type;
    private final byte[] data;

    private Tensor(String name, long[] shape, int type, byte[] data) {
        this.name = Objects.requireNonNull(name, "name");
        this.shape = Objects.requireNonNull(shape, "shape").clone();
        this.type = type;
        this.data = Objects.requireNonNull(data, "data").clone();
        if (name.isEmpty() || (type != 0 && type != 1)) throw new IllegalArgumentException("Invalid tensor specification");
        long bytes = type == 0 ? 4 : 8;
        for (long dimension : this.shape) {
            if (dimension <= 0 || dimension > Integer.MAX_VALUE) throw new IllegalArgumentException("Invalid static shape");
            bytes = Math.multiplyExact(bytes, dimension);
        }
        if (bytes != data.length) throw new IllegalArgumentException("Tensor byte count differs from shape");
    }
    public static Tensor float32(String name, long[] shape, float[] values) {
        Objects.requireNonNull(values, "values");
        ByteBuffer buffer = ByteBuffer.allocate(Math.multiplyExact(values.length, 4)).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(values);
        return new Tensor(name, shape, 0, buffer.array());
    }
    public static Tensor int64(String name, long[] shape, long[] values) {
        Objects.requireNonNull(values, "values");
        ByteBuffer buffer = ByteBuffer.allocate(Math.multiplyExact(values.length, 8)).order(ByteOrder.nativeOrder());
        buffer.asLongBuffer().put(values);
        return new Tensor(name, shape, 1, buffer.array());
    }
    public String name() { return name; }
    public long[] shape() { return shape.clone(); }
    public DType dtype() { return type == 0 ? DType.FLOAT32 : DType.INT64; }
    public float[] floats() {
        if (type != 0) throw new IllegalStateException("Expected FP32 tensor");
        float[] result = new float[data.length / 4];
        ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asFloatBuffer().get(result);
        return result;
    }
    public long[] longs() {
        if (type != 1) throw new IllegalStateException("Expected INT64 tensor");
        long[] result = new long[data.length / 8];
        ByteBuffer.wrap(data).order(ByteOrder.nativeOrder()).asLongBuffer().get(result);
        return result;
    }
}
