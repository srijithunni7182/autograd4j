package com.autograd;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Arrays;

/**
 * MicroGPT — A complete GPT language model in pure Java, no dependencies.
 * <p>
 * Faithful port of Andrej Karpathy's microgpt.py art project:
 * https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 * And Milan's C# port: https://github.com/milanm/AutoGrad-Engine
 * </p>
 * <p>
 * This is the exact same algorithm behind ChatGPT, stripped to its essence.
 * Real GPTs have billions of parameters and run on GPU clusters.
 * This one has ~5,000 parameters and runs on a single CPU thread.
 * But every conceptual piece is here. Everything else is "just" optimization.
 * </p>
 */

public class MicroGPT {

    /** Private constructor to prevent instantiation. */
    private MicroGPT() {
    }

    // -------------------------------------------------------------------------
    // HYPERPARAMETERS
    // -------------------------------------------------------------------------
    /** Embedding dimension (vector size). default: 16. */
    static int nEmbd;
    /** Number of transformer layers. default: 1. */
    static int nLayer;
    /** Maximum sequence length (context window). default: 8. */
    static int blockSize;
    /** Number of training steps. default: 1000. */
    static int numSteps;
    /** Number of attention heads. default: 4. */
    static int nHead;
    /** Learning rate for Adam optimizer. default: 1e-2. */
    static double learningRate;
    /** Random seed for reproducibility. default: 42. */
    static long seed;
    /** Dimension of each attention head (nEmbd / nHead). */
    static int headDim;
    /** Whether to use the gated attention mechanism. default: false. */
    static boolean useGated = false;

    /** The tokenizer instance for encoding/decoding text. */
    static Tokenizer tokenizer;

    // -------------------------------------------------------------------------
    // MODEL WEIGHTS
    // -------------------------------------------------------------------------
    /**
     * The state dictionary holding all model weights (matrices).
     * Key: parameter name (e.g. "wte", "layer0.attn_wq").
     * Value: 2D array of Values.
     */
    static Map<String, Value[][]> stateDict = new HashMap<>();

    /** Flattened list of all parameters for the optimizer. */
    static List<Value> parameters = new ArrayList<>();

    /** Random number generator. */
    static Random rng;

    /**
     * Helper to parse a boolean command-line argument.
     * 
     * @param args       Command line arguments.
     * @param name       Argument name (without --).
     * @param defaultVal Default value if not found.
     * @return Parsed boolean.
     */
    static boolean parseArg(String[] args, String name, boolean defaultVal) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--" + name)) {
                int val = Integer.parseInt(args[i + 1]);
                return val != 0;
            }
        }
        return defaultVal;
    }

    /**
     * Helper to parse an integer command-line argument.
     * 
     * @param args       Command line arguments.
     * @param name       Argument name (without --).
     * @param defaultVal Default value if not found.
     * @return Parsed integer.
     */
    static int parseArg(String[] args, String name, int defaultVal) {
        for (int i = 0; i < args.length - 1; i++)
            if (args[i].equals("--" + name))
                return Integer.parseInt(args[i + 1]);
        return defaultVal;
    }

    /**
     * Helper to parse a double command-line argument.
     * 
     * @param args       Command line arguments.
     * @param name       Argument name (without --).
     * @param defaultVal Default value if not found.
     * @return Parsed double.
     */
    static double parseArg(String[] args, String name, double defaultVal) {
        for (int i = 0; i < args.length - 1; i++)
            if (args[i].equals("--" + name))
                return Double.parseDouble(args[i + 1]);
        return defaultVal;
    }

    /**
     * Helper to parse a long command-line argument.
     * 
     * @param args       Command line arguments.
     * @param name       Argument name (without --).
     * @param defaultVal Default value if not found.
     * @return Parsed long.
     */
    static long parseArg(String[] args, String name, long defaultVal) {
        for (int i = 0; i < args.length - 1; i++)
            if (args[i].equals("--" + name))
                return Long.parseLong(args[i + 1]);
        return defaultVal;
    }

    /**
     * Initialize matrix with random values (Gaussian distribution).
     * 
     * @param nout Number of output rows.
     * @param nin  Number of input columns.
     * @param std  Standard deviation for initialization.
     * @return Initialized matrix of Values.
     */
    static Value[][] matrix(int nout, int nin, double std) {
        Value[][] mat = new Value[nout][nin];
        for (int i = 0; i < nout; i++) {
            for (int j = 0; j < nin; j++) {
                mat[i][j] = new Value(rng.nextGaussian() * std);
            }
        }
        return mat;
    }

    /**
     * The GPT Forward Pass.
     * Builds the computation graph for a single token's prediction.
     *
     * @param tokenId Current token ID.
     * @param posId   Current position ID.
     * @param keys    KV Capture for Keys (for attention).
     * @param values  KV Capture for Values (for attention).
     * @return Logits (raw scores) for the next token.
     */
    static Value[] gpt(int tokenId, int posId, List<List<Value[]>> keys, List<List<Value[]>> values) {
        // Step 1: Look up embeddings
        Value[] tokEmb = stateDict.get("wte")[tokenId];
        Value[] posEmb = stateDict.get("wpe")[posId % blockSize];

        Value[] x = new Value[tokEmb.length];
        for (int i = 0; i < tokEmb.length; i++) {
            x[i] = tokEmb[i].add(posEmb[i]);
        }

        for (int li = 0; li < nLayer; li++) {
            // =================================================================
            // MULTI-HEAD SELF-ATTENTION
            // =================================================================

            Value[] xResidual = x; // save for residual connection
            x = NeuralOps.rmsNorm(x);

            // Project input into Q, K, V
            Value[] q = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] val = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wv"));

            // Store K and V in the cache
            keys.get(li).add(k);
            values.get(li).add(val);

            List<Value> xAttn = new ArrayList<>();
            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;

                // Slice q for this head
                Value[] qH = Arrays.copyOfRange(q, hs, hs + headDim);

                // Slice all past k's for this head
                List<Value[]> kH = new ArrayList<>();
                for (Value[] ki : keys.get(li)) {
                    kH.add(Arrays.copyOfRange(ki, hs, hs + headDim));
                }

                // Slice all past v's for this head
                List<Value[]> vH = new ArrayList<>();
                for (Value[] vi : values.get(li)) {
                    vH.add(Arrays.copyOfRange(vi, hs, hs + headDim));
                }

                // Compute attention scores
                Value[] attnLogits = new Value[kH.size()];
                double scale = Math.sqrt(headDim);

                for (int t = 0; t < kH.size(); t++) {
                    Value dot = new Value(0);
                    for (int j = 0; j < headDim; j++) {
                        dot = dot.add(qH[j].mul(kH.get(t)[j]));
                    }
                    attnLogits[t] = dot.div(scale);
                }

                // Softmax
                Value[] attnWeights = NeuralOps.softmax(attnLogits);

                // Weighted blend
                for (int j = 0; j < headDim; j++) {
                    Value sum = new Value(0);
                    for (int t = 0; t < vH.size(); t++) {
                        sum = sum.add(attnWeights[t].mul(vH.get(t)[j]));
                    }
                    xAttn.add(sum);
                }
            }

            // Combine heads and project
            x = NeuralOps.linear(xAttn.toArray(new Value[0]), stateDict.get("layer" + li + ".attn_wo"));

            // Residual connection
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].add(xResidual[i]);

            // =================================================================
            // MLP (FEED-FORWARD NETWORK)
            // =================================================================
            xResidual = x;
            x = NeuralOps.rmsNorm(x);
            x = NeuralOps.linear(x, stateDict.get("layer" + li + ".mlp_fc1")); // expand

            // Squared ReLU
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].relu().pow(2);

            x = NeuralOps.linear(x, stateDict.get("layer" + li + ".mlp_fc2")); // compress

            // Residual connection
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].add(xResidual[i]);
        }

        // Final prediction
        return NeuralOps.linear(x, stateDict.get("wte")); // Weight tying
    }

    /**
     * The GPT Forward Pass with Gated Attention.
     *
     * @param tokenId Current token ID.
     * @param posId   Current position ID.
     * @param keys    KV Capture for Keys.
     * @param values  KV Capture for Values.
     * @return Logits (raw scores) for the next token.
     */
    static Value[] gptGated(int tokenId, int posId, List<List<Value[]>> keys, List<List<Value[]>> values) {
        // Step 1: Look up embeddings
        Value[] tokEmb = stateDict.get("wte")[tokenId];
        Value[] posEmb = stateDict.get("wpe")[posId % blockSize];

        Value[] x = new Value[tokEmb.length];
        for (int i = 0; i < tokEmb.length; i++) {
            x[i] = tokEmb[i].add(posEmb[i]);
        }

        for (int li = 0; li < nLayer; li++) {
            // =================================================================
            // MULTI-HEAD SELF-ATTENTION
            // =================================================================

            Value[] xResidual = x; // save for residual connection
            x = NeuralOps.rmsNorm(x);

            // Calculate Gate based on normalized input
            Value[] gateRaw = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_gate"));
            Value[] gate = new Value[gateRaw.length];
            for (int i = 0; i < gateRaw.length; i++)
                gate[i] = gateRaw[i].sigmoid();

            // Project input into Q, K, V
            Value[] q = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] val = NeuralOps.linear(x, stateDict.get("layer" + li + ".attn_wv"));

            // Store K and V in the cache
            keys.get(li).add(k);
            values.get(li).add(val);

            List<Value> xAttn = new ArrayList<>();
            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;

                // Slice q for this head
                Value[] qH = Arrays.copyOfRange(q, hs, hs + headDim);

                // Slice all past k's for this head
                List<Value[]> kH = new ArrayList<>();
                for (Value[] ki : keys.get(li)) {
                    kH.add(Arrays.copyOfRange(ki, hs, hs + headDim));
                }

                // Slice all past v's for this head
                List<Value[]> vH = new ArrayList<>();
                for (Value[] vi : values.get(li)) {
                    vH.add(Arrays.copyOfRange(vi, hs, hs + headDim));
                }

                // Compute attention scores
                Value[] attnLogits = new Value[kH.size()];
                double scale = Math.sqrt(headDim);

                for (int t = 0; t < kH.size(); t++) {
                    Value dot = new Value(0);
                    for (int j = 0; j < headDim; j++) {
                        dot = dot.add(qH[j].mul(kH.get(t)[j]));
                    }
                    attnLogits[t] = dot.div(scale);
                }

                // Softmax
                Value[] attnWeights = NeuralOps.softmax(attnLogits);

                // Weighted blend
                for (int j = 0; j < headDim; j++) {
                    Value sum = new Value(0);
                    for (int t = 0; t < vH.size(); t++) {
                        sum = sum.add(attnWeights[t].mul(vH.get(t)[j]));
                    }
                    xAttn.add(sum);
                }
            }

            // Combine heads and project
            x = NeuralOps.linear(xAttn.toArray(new Value[0]), stateDict.get("layer" + li + ".attn_wo"));

            // Apply Gate
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].mul(gate[i]);
            }

            // Residual connection
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].add(xResidual[i]);

            // =================================================================
            // MLP (FEED-FORWARD NETWORK)
            // =================================================================
            xResidual = x;
            x = NeuralOps.rmsNorm(x);
            x = NeuralOps.linear(x, stateDict.get("layer" + li + ".mlp_fc1")); // expand

            // Squared ReLU
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].relu().pow(2);

            x = NeuralOps.linear(x, stateDict.get("layer" + li + ".mlp_fc2")); // compress

            // Residual connection
            for (int i = 0; i < x.length; i++)
                x[i] = x[i].add(xResidual[i]);
        }

        // Final prediction
        return NeuralOps.linear(x, stateDict.get("wte")); // Weight tying
    }

    /**
     * Main entry point.
     * Configures, trains, and runs the MicroGPT model.
     *
     * @param args Command line arguments.
     * @throws IOException          If dataset download fails.
     * @throws InterruptedException If dataset download is interrupted.
     */
    public static void main(String[] args) throws IOException, InterruptedException {
        // Parse CLI arguments
        nEmbd = parseArg(args, "n_embd", 16);
        nLayer = parseArg(args, "n_layer", 1);
        blockSize = parseArg(args, "block_size", 8);
        numSteps = parseArg(args, "num_steps", 1000);
        nHead = parseArg(args, "n_head", 4);
        learningRate = parseArg(args, "learning_rate", 1e-2);
        seed = parseArg(args, "seed", 42L);
        rng = new Random(seed);
        headDim = nEmbd / nHead;
        useGated = parseArg(args, "gated", false);

        if (useGated) {
            System.out.println("Using Gated Attention Mechanism");
        }

        // ---------------------------------------------------------------------
        // DATASET
        // ---------------------------------------------------------------------
        String inputFile = "input.txt";
        File f = new File(inputFile);
        if (!f.exists()) {
            System.out.println("Downloading names dataset...");
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"))
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            Files.writeString(Path.of(inputFile), response.body());
        }

        List<String> allLines = Files.readAllLines(Path.of(inputFile));
        List<String> docs = allLines.stream()
                .map(String::trim)
                .filter(l -> !l.isEmpty())
                .collect(Collectors.toList());

        // Shuffle
        Collections.shuffle(docs, rng);

        // Tokenizer
        tokenizer = new Tokenizer(docs);
        System.out.println("vocab size: " + tokenizer.vocabSize + ", num docs: " + docs.size());

        // ---------------------------------------------------------------------
        // MODEL PARAMETERS
        // ---------------------------------------------------------------------
        stateDict.put("wte", matrix(tokenizer.vocabSize, nEmbd, 0.02));
        stateDict.put("wpe", matrix(blockSize, nEmbd, 0.02));

        for (int i = 0; i < nLayer; i++) {
            stateDict.put("layer" + i + ".attn_wq", matrix(nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".attn_wk", matrix(nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".attn_wv", matrix(nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".attn_wo", matrix(nEmbd, nEmbd, 0.0)); // zero init
            stateDict.put("layer" + i + ".mlp_fc1", matrix(4 * nEmbd, nEmbd, 0.02));
            stateDict.put("layer" + i + ".mlp_fc2", matrix(nEmbd, 4 * nEmbd, 0.0)); // zero init
            if (useGated) {
                stateDict.put("layer" + i + ".attn_gate", matrix(nEmbd, nEmbd, 0.02));
            }
        }

        // Collect parameters in deterministic order
        List<String> paramKeys = new ArrayList<>();
        paramKeys.add("wte");
        paramKeys.add("wpe");
        for (int i = 0; i < nLayer; i++) {
            paramKeys.add("layer" + i + ".attn_wq");
            paramKeys.add("layer" + i + ".attn_wk");
            paramKeys.add("layer" + i + ".attn_wv");
            paramKeys.add("layer" + i + ".attn_wo");
            paramKeys.add("layer" + i + ".mlp_fc1");
            paramKeys.add("layer" + i + ".mlp_fc2");
            if (useGated) {
                paramKeys.add("layer" + i + ".attn_gate");
            }
        }

        for (String key : paramKeys) {
            Value[][] mat = stateDict.get(key);
            for (Value[] row : mat) {
                for (Value v : row) {
                    parameters.add(v);
                }
            }
        }
        System.out.println("num params: " + parameters.size());

        train(docs);
        generate(5);
    }

    /**
     * Training Loop.
     * Runs the forward pass, calculates loss, performs backpropagation, and updates
     * weights.
     *
     * @param docs List of training documents.
     */
    static void train(List<String> docs) {
        // Adam optimizer state
        double beta1 = 0.9, beta2 = 0.95, epsAdam = 1e-8;
        double[] mState = new double[parameters.size()];
        double[] vState = new double[parameters.size()];

        for (int step = 0; step < numSteps; step++) {
            String doc = docs.get(step % docs.size());
            List<Integer> tokens = tokenizer.encode(doc);
            if (tokens.size() > blockSize) {
                tokens = tokens.subList(0, blockSize);
            }

            // Init KV Caches
            List<List<Value[]>> keys = new ArrayList<>();
            List<List<Value[]>> values = new ArrayList<>();
            for (int i = 0; i < nLayer; i++) {
                keys.add(new ArrayList<>());
                values.add(new ArrayList<>());
            }

            double lossF = 0.0;

            // Forward pass
            for (int posId = 0; posId < tokens.size() - 1; posId++) {
                Value[] logits;
                if (useGated) {
                    logits = gptGated(tokens.get(posId), posId, keys, values);
                } else {
                    logits = gpt(tokens.get(posId), posId, keys, values);
                }
                Value[] probs = NeuralOps.softmax(logits);

                // Cross-entropy loss
                int targetToken = tokens.get(posId + 1);
                Value loss = probs[targetToken].log().neg();
                loss = loss.mul(1.0 / (tokens.size() - 1)); // average

                loss.backward();
                lossF += loss.data;
            }

            // Adam Update
            double lrT = learningRate * (1.0 - (double) step / numSteps);
            for (int i = 0; i < parameters.size(); i++) {
                Value p = parameters.get(i);

                mState[i] = beta1 * mState[i] + (1 - beta1) * p.grad;
                vState[i] = beta2 * vState[i] + (1 - beta2) * p.grad * p.grad;

                double mHat = mState[i] / (1 - Math.pow(beta1, step + 1));
                double vHat = vState[i] / (1 - Math.pow(beta2, step + 1));

                p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
                p.grad = 0; // zero grad
            }

            System.out.printf("step %d / %d | loss %.4f%n", step + 1, numSteps, lossF);
        }
    }

    /**
     * Generation Loop.
     * Generates new text by sampling from the model.
     *
     * @param numSamples Number of samples to generate.
     */
    static void generate(int numSamples) {
        System.out.println("\n--- generation ---");
        for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
            List<List<Value[]>> keys = new ArrayList<>();
            List<List<Value[]>> values = new ArrayList<>();
            for (int i = 0; i < nLayer; i++) {
                keys.add(new ArrayList<>());
                values.add(new ArrayList<>());
            }

            int tokenId = tokenizer.BOS;
            StringBuilder generated = new StringBuilder();

            for (int posId = 0; posId < blockSize; posId++) {
                Value[] logits;
                if (useGated) {
                    logits = gptGated(tokenId, posId, keys, values);
                } else {
                    logits = gpt(tokenId, posId, keys, values);
                }
                Value[] probs = NeuralOps.softmax(logits);

                // Sample next token
                double r = rng.nextDouble();
                double cumulative = 0;
                tokenId = 0;
                for (int i = 0; i < probs.length; i++) {
                    cumulative += probs[i].data;
                    if (r <= cumulative) {
                        tokenId = i;
                        break;
                    }
                }

                if (tokenId == tokenizer.EOS)
                    break;
                generated.append(tokenizer.decode(tokenId));
            }

            System.out.println("sample " + sampleIdx + ": " + generated.toString());
        }
    }
}
