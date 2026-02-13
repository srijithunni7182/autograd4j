package com.autograd;

import java.util.ArrayList;
import java.util.List;

public class MicroGPTTest {

    public static void runTests() {
        testTokenizer();
        testModelInitialization();
        System.out.println("MicroGPTTest: All tests passed!");
    }

    private static void testTokenizer() {
        List<String> docs = new ArrayList<>();
        docs.add("aba");
        Tokenizer t = new Tokenizer(docs);

        // Vocab should be <BOS>, <EOS>, a, b. Size 4.
        if (t.vocabSize != 4)
            throw new RuntimeException("Tokenizer vocab size unexpected: " + t.vocabSize);

        List<Integer> encoded = t.encode("aba");
        // <BOS>, a, b, a, <EOS> -> 0, 2, 3, 2, 1 (assuming sorted order a,b)
        // logic: <BOS>=0, <EOS>=1. 'a' is index 2, 'b' is index 3.

        if (encoded.size() != 5)
            throw new RuntimeException("Encoded size unexpected");
        if (encoded.get(0) != t.BOS)
            throw new RuntimeException("First token not BOS");
        if (encoded.get(encoded.size() - 1) != t.EOS)
            throw new RuntimeException("Last token not EOS");

        String decoded = t.decode(encoded.get(1)); // should be 'a'
        if (!decoded.equals("a"))
            throw new RuntimeException("Decoding failed");

        System.out.println("  testTokenizer passed");
    }

    private static void testModelInitialization() {
        // Just verify we can set up the model without crashing
        MicroGPT.nEmbd = 16;
        MicroGPT.nLayer = 2;
        MicroGPT.nHead = 2;
        MicroGPT.blockSize = 8;
        MicroGPT.headDim = 8; // 16/2

        // This is a partial test since detailed model verification is hard without
        // running the full loops,
        // but verifying the NeuralOps work on dummy data is good.
        Value[] input = new Value[16];
        for (int i = 0; i < 16; i++)
            input[i] = new Value(0.5);

        Value[] output = NeuralOps.rmsNorm(input);
        if (output.length != 16)
            throw new RuntimeException("RMSNorm output size wrong");

        System.out.println("  testModelInitialization passed");
    }
}
