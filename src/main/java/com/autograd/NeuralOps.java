package com.autograd;

import java.util.Arrays;

/**
 * Neural Ops — stateless building blocks used by the GPT model.
 */

public class NeuralOps {

    /** Private constructor to prevent instantiation. */
    private NeuralOps() {
    }

    /**
     * Linear: Matrix-vector multiplication. The fundamental operation of neural
     * nets.
     * <p>
     * Takes an input vector x and a weight matrix w, returns w * x.
     * Each output element is a weighted sum of all inputs — the weights decide
     * which inputs matter and how much.
     * </p>
     * <p>
     * Example: if x = [0.5, 0.3] and w = [[2, 1], [0, 3]], then:
     * output[0] = 2*0.5 + 1*0.3 = 1.3
     * output[1] = 0*0.5 + 3*0.3 = 0.9
     * </p>
     * <p>
     * The magic: these weights are learned during training. The model discovers
     * which combinations of input features are useful for prediction.
     * </p>
     *
     * @param x Input vector.
     * @param w Weight matrix.
     * @return Output vector (result of w * x).
     */
    public static Value[] linear(Value[] x, Value[][] w) {
        Value[] result = new Value[w.length];
        for (int o = 0; o < w.length; o++) {
            Value sum = new Value(0);
            for (int i = 0; i < x.length; i++) {
                sum = sum.add(w[o][i].mul(x[i]));
            }
            result[o] = sum;
        }
        return result;
    }

    /**
     * Softmax: Converts raw scores into probabilities that sum to 1.
     * <p>
     * Input: [2.0, 1.0, 0.1] (raw scores, called "logits")
     * Output: [0.66, 0.24, 0.10] (probabilities)
     * </p>
     * <p>
     * Higher scores get higher probabilities. The exponential makes big
     * differences more extreme — if one score is much higher, it dominates.
     * </p>
     * <p>
     * The "subtract max" trick prevents numerical overflow. e^1000 is infinity,
     * but e^0 is fine. Subtracting the max doesn't change the relative
     * probabilities (it cancels out in the division).
     * </p>
     *
     * @param logits Raw score vector.
     * @return Probability vector.
     */
    public static Value[] softmax(Value[] logits) {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (Value v : logits) {
            if (v.data > maxVal) {
                maxVal = v.data;
            }
        }

        final double finalMax = maxVal;
        Value[] exps = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = logits[i].sub(finalMax).exp();
        }

        Value total = new Value(0);
        for (Value e : exps) {
            total = total.add(e);
        }

        Value[] out = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = exps[i].div(total);
        }

        return out;
    }

    /**
     * RMSNorm (Root Mean Square Normalization): Keeps numbers in a healthy range.
     * <p>
     * Without normalization, values can grow or shrink as they pass through layers,
     * making training unstable. RMSNorm scales the vector so its average squared
     * magnitude is ~1. Think of it like auto-adjusting the volume.
     * </p>
     * <p>
     * This is a simpler alternative to LayerNorm (used in the original GPT-2).
     * LLaMA and most modern models use RMSNorm because it works just as well
     * with fewer operations.
     * </p>
     *
     * @param x Input vector.
     * @return Normalized vector.
     */
    public static Value[] rmsNorm(Value[] x) {
        Value ms = new Value(0);
        for (Value xi : x) {
            ms = ms.add(xi.mul(xi));
        }
        ms = ms.div(x.length);

        // 1e-5 prevents division by zero
        Value scale = ms.add(1e-5).pow(-0.5);

        Value[] out = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].mul(scale);
        }
        return out;
    }
}
