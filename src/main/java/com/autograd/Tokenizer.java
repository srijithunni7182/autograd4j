package com.autograd;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Tokenizer — converts between characters and numbers.
 * <p>
 * Neural networks only understand numbers. So we assign each character an ID:
 * &lt;BOS&gt; = 0 (Beginning Of Sequence — signals "start of a name")
 * &lt;EOS&gt; = 1 (End Of Sequence — signals "this name is done")
 * 'a' = 2, 'b' = 3, ... etc.
 * </p>
 * <p>
 * "emma" becomes: [0, 6, 14, 14, 2, 1] (BOS, e, m, m, a, EOS)
 * </p>
 */

public class Tokenizer {
    /** List of all unique characters in the vocabulary. */
    public final List<String> chars;
    /** string-to-integer lookup */
    public final Map<String, Integer> stoi;
    /** integer-to-string lookup */
    public final Map<Integer, String> itos;
    /** Size of the vocabulary. */
    public final int vocabSize;
    /** Beginning of Sequence token ID. */
    public final int BOS;
    /** End of Sequence token ID. */
    public final int EOS;

    /**
     * Initializes the tokenizer from a list of strings (documents).
     * 
     * @param docs List of training documents (names).
     */
    public Tokenizer(List<String> docs) {
        // Collect every unique character from all names, plus BOS and EOS markers.
        Set<String> allChars = new TreeSet<>();
        for (String doc : docs) {
            for (char c : doc.toCharArray()) {
                allChars.add(String.valueOf(c));
            }
        }

        chars = new ArrayList<>();
        chars.add("<BOS>");
        chars.add("<EOS>");
        chars.addAll(allChars);

        vocabSize = chars.size();
        stoi = new HashMap<>();
        itos = new HashMap<>();

        for (int i = 0; i < chars.size(); i++) {
            stoi.put(chars.get(i), i);
            itos.put(i, chars.get(i));
        }

        BOS = stoi.get("<BOS>");
        EOS = stoi.get("<EOS>");
    }

    /**
     * Encode a string into a list of token IDs, with BOS/EOS markers.
     *
     * @param text The input string to encode.
     * @return A list of integer token IDs including BOS and EOS.
     */
    public List<Integer> encode(String text) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(BOS);
        for (char c : text.toCharArray()) {
            tokens.add(stoi.get(String.valueOf(c)));
        }
        tokens.add(EOS);
        return tokens;
    }

    /**
     * Decode a token ID back to its string representation.
     *
     * @param tokenId The integer token ID.
     * @return The corresponding string character (or special token).
     */
    public String decode(int tokenId) {
        return itos.get(tokenId);
    }
}
