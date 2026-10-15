package org.xrgpu;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.*;

public final class WordPiece {
    private final Map<String, Integer> vocab = new HashMap<>();
    private static final Pattern SPECIAL = Pattern.compile("\\[(?:UNK|SEP|PAD|CLS|MASK)\\]");
    public static final class Tokens {
        public final long[] ids, mask;
        Tokens(int size) { ids = new long[size]; mask = new long[size]; }
    }
    public WordPiece(InputStream input) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8))) {
            String line; int i = 0;
            while ((line = reader.readLine()) != null) vocab.put(line, i++);
        }
        for (String name : new String[]{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"})
            if (!vocab.containsKey(name)) throw new IOException("Missing token " + name);
    }
    private static boolean chinese(int c) {
        return (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) ||
            (c >= 0x20000 && c <= 0x2a6df) || (c >= 0x2a700 && c <= 0x2b73f) ||
            (c >= 0x2b740 && c <= 0x2b81f) || (c >= 0x2b820 && c <= 0x2ceaf) ||
            (c >= 0xf900 && c <= 0xfaff) || (c >= 0x2f800 && c <= 0x2fa1f);
    }
    private static boolean punctuation(int c) {
        int t = Character.getType(c);
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
            (c >= 123 && c <= 126) || t == Character.CONNECTOR_PUNCTUATION ||
            t == Character.DASH_PUNCTUATION || t == Character.START_PUNCTUATION ||
            t == Character.END_PUNCTUATION || t == Character.OTHER_PUNCTUATION ||
            t == Character.INITIAL_QUOTE_PUNCTUATION || t == Character.FINAL_QUOTE_PUNCTUATION;
    }
    private void word(String token, List<Integer> result) {
        if (token.isEmpty()) return;
        if (token.codePointCount(0, token.length()) > 100) { result.add(vocab.get("[UNK]")); return; }
        ArrayList<Integer> pieces = new ArrayList<>();
        int start = 0;
        while (start < token.length()) {
            int end = token.length(); Integer found = null;
            while (end > start) {
                found = vocab.get((start == 0 ? "" : "##") + token.substring(start, end));
                if (found != null) break;
                end = token.offsetByCodePoints(end, -1);
            }
            if (found == null) { result.add(vocab.get("[UNK]")); return; }
            pieces.add(found); start = end;
        }
        result.addAll(pieces);
    }
    private void ordinary(String source, List<Integer> output) {
        StringBuilder clean = new StringBuilder();
        source.codePoints().forEach(c -> {
            int t = Character.getType(c);
            if (c == 0 || c == 0xfffd) return;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || t == Character.SPACE_SEPARATOR) clean.append(' ');
            else if (t != Character.CONTROL && t != Character.FORMAT) {
                if (chinese(c)) clean.append(' ');
                clean.appendCodePoint(c);
                if (chinese(c)) clean.append(' ');
            }
        });
        String normalized = Normalizer.normalize(clean.toString().toLowerCase(Locale.ROOT), Normalizer.Form.NFD);
        StringBuilder token = new StringBuilder();
        normalized.codePoints().forEach(c -> {
            if (Character.getType(c) == Character.NON_SPACING_MARK) return;
            if (Character.isWhitespace(c) || punctuation(c)) {
                word(token.toString(), output); token.setLength(0);
                if (punctuation(c)) word(new String(Character.toChars(c)), output);
            } else token.appendCodePoint(c);
        });
        word(token.toString(), output);
    }
    public Tokens encode(String text, int length) {
        if (length < 2 || length > 256) throw new IllegalArgumentException("length must be 2..256");
        List<Integer> ids = new ArrayList<>();
        Matcher matcher = SPECIAL.matcher(text); int position = 0;
        while (matcher.find()) {
            ordinary(text.substring(position, matcher.start()), ids);
            ids.add(vocab.get(matcher.group())); position = matcher.end();
        }
        ordinary(text.substring(position), ids);
        Tokens output = new Tokens(length);
        Arrays.fill(output.ids, vocab.get("[PAD]"));
        output.ids[0] = vocab.get("[CLS]"); output.mask[0] = 1;
        int used = Math.min(ids.size(), length - 2);
        for (int i = 0; i < used; ++i) { output.ids[i + 1] = ids.get(i); output.mask[i + 1] = 1; }
        output.ids[used + 1] = vocab.get("[SEP]"); output.mask[used + 1] = 1;
        return output;
    }
}
