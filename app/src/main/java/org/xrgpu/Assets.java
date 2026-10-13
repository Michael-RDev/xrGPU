package org.xrgpu;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Iterator;

public final class Assets {
    public final File root;
    public final JSONObject manifest;
    public final WordPiece tokenizer;
    public Assets(Context context) throws Exception {
        root = new File(context.getFilesDir(), "bundle"); root.mkdirs();
        String manifestText;
        try (InputStream in = context.getAssets().open("bundle/bundle.json")) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream(); copy(in, bytes);
            manifestText = bytes.toString("UTF-8");
        }
        manifest = new JSONObject(manifestText);
        if (manifest.getBoolean("references_only")) throw new IOException("Model bundle is incomplete");
        JSONObject files = manifest.getJSONObject("files");
        for (Iterator<String> keys = files.keys(); keys.hasNext();) {
            String name = keys.next(); File target = new File(root, name);
            if (!target.getCanonicalPath().startsWith(root.getCanonicalPath() + File.separator))
                throw new IOException("Invalid asset path");
            String expected = files.getString(name);
            if (target.isFile() && digest(target).equals(expected)) continue;
            target.getParentFile().mkdirs();
            File temporary = new File(target.getPath() + ".part");
            try (InputStream in = context.getAssets().open("bundle/" + name);
                 OutputStream out = new FileOutputStream(temporary)) { copy(in, out); }
            if (!digest(temporary).equals(expected)) throw new IOException("Asset checksum mismatch: " + name);
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        tokenizer = new WordPiece(new FileInputStream(new File(root, "tokenizer/vocab.txt")));
    }
    public String text(String name) throws IOException {
        return new String(Files.readAllBytes(new File(root, name).toPath()), StandardCharsets.UTF_8);
    }
    public String model(String backend, int length) { return new File(root, "minilm_" + backend + "_" + length + ".pte").getPath(); }
    public static String digest(File file) throws Exception {
        MessageDigest d = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(file)) {
            byte[] block = new byte[1024 * 1024]; int n;
            while ((n = in.read(block)) != -1) d.update(block, 0, n);
        }
        StringBuilder out = new StringBuilder();
        for (byte b : d.digest()) out.append(String.format("%02x", b & 255));
        return out.toString();
    }
    private static void copy(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[65536]; int count;
        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
    }
    public float[] floats(String name) throws IOException {
        byte[] bytes = Files.readAllBytes(new File(root, name).toPath());
        if (bytes.length % 4 != 0) throw new IOException("Invalid float fixture");
        float[] output = new float[bytes.length / 4];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(output); return output;
    }
    public static void writeAtomic(File file, JSONObject value) throws Exception {
        file.getParentFile().mkdirs();
        File temp = new File(file.getPath() + ".part");
        Files.write(temp.toPath(), value.toString(2).getBytes(StandardCharsets.UTF_8));
        Files.move(temp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }
}
