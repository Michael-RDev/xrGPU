package org.xrgpu;
import java.util.Arrays;
public final class LibraryTensorTests {
    private static int checks = 0;
    private static void check(boolean value) { checks++; if (!value) throw new AssertionError("check " + checks); }
    private static void invalid(Runnable runnable) {
        checks++;
        try { runnable.run(); throw new AssertionError("Expected failure"); }
        catch (IllegalArgumentException | ArithmeticException | NullPointerException expected) { }
    }
    public static void main(String[] args) {
        long[] shape = {1, 3}; float[] values = {1, -2, 3.5f};
        Tensor tensor = Tensor.float32("input", shape, values);
        shape[0] = 8; values[0] = 50;
        check(tensor.name().equals("input"));
        check(tensor.dtype() == Tensor.DType.FLOAT32);
        check(Arrays.equals(tensor.shape(), new long[]{1, 3}));
        check(Arrays.equals(tensor.floats(), new float[]{1, -2, 3.5f}));
        float[] output = tensor.floats(); output[0] = 99;
        long[] outputShape = tensor.shape(); outputShape[0] = 77;
        check(tensor.floats()[0] == 1); check(tensor.shape()[0] == 1);
        Tensor tokens = Tensor.int64("tokens", new long[]{2}, new long[]{Long.MAX_VALUE, Long.MIN_VALUE});
        check(tokens.dtype() == Tensor.DType.INT64);
        check(Arrays.equals(tokens.longs(), new long[]{Long.MAX_VALUE, Long.MIN_VALUE}));
        try { tensor.longs(); throw new AssertionError("Expected type failure"); }
        catch (IllegalStateException expected) { checks++; }
        try { tokens.floats(); throw new AssertionError("Expected type failure"); }
        catch (IllegalStateException expected) { checks++; }
        invalid(() -> Tensor.float32("bad", new long[]{0}, new float[]{}));
        invalid(() -> Tensor.float32("bad", new long[]{-1}, new float[]{}));
        invalid(() -> Tensor.float32("bad", new long[]{Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE}, new float[]{}));
        invalid(() -> Tensor.float32("bad", new long[]{2}, new float[]{1}));
        invalid(() -> Tensor.float32("", new long[]{1}, new float[]{1}));
        invalid(() -> Tensor.int64("bad", null, new long[]{1}));
        check(Tensor.float32("scalar", new long[]{}, new float[]{7}).floats()[0] == 7);
        System.out.println("Library tensor checks passed: " + checks);
    }
}
