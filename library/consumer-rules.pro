# JNI accesses these immutable tensor fields and the private output constructor.
-keep class org.xrgpu.Tensor { *; }
-keep class org.xrgpu.Session { *; }
-keep class org.xrgpu.Session$Options { *; }
