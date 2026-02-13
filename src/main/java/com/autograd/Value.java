package com.autograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.function.Consumer;

/**
 * <h2>The Autograd Engine</h2>
 * <p>
 * The big question in neural networks: "If I change this one number a tiny bit,
 * how much does the overall error change?" The answer is called a "gradient."
 * </p>
 * <p>
 * The Value class wraps every number and secretly records every math operation
 * done to it — building a chain. When you call Backward(), it walks backward
 * through that chain to figure out each number's influence on the final error.
 * </p>
 * <p>
 * This technique is called "backpropagation." It's the core of how neural
 * networks learn. In production, PyTorch or TensorFlow does this on tensors
 * (big arrays of numbers) on GPUs. Here we do it one scalar at a time.
 * </p>
 * <p>
 * Think of it like a spreadsheet: if cell Z depends on Y, which depends on X,
 * and you change X by 0.001, how much does Z change? The chain rule from
 * calculus answers this automatically for any chain of operations.
 * </p>
 */

public class Value {
    /** The actual number (e.g. 0.37, -1.2, etc.) */
    public double data;

    /**
     * The gradient: "how much does the final error change if I nudge this number?"
     * Starts at 0, gets filled in during backward().
     */
    public double grad;

    /**
     * A runnable that computes this node's contribution to gradients.
     * Each operation (+, *, exp, etc.) defines its own gradient rule.
     */
    private Runnable _backward;

    /**
     * The Values that were inputs to the operation that created this Value.
     * This is how we track the computation graph — like a family tree of math.
     */
    private final Set<Value> _prev;

    /** What operation created this node ("+", "*", "exp", etc.) — for debugging. */
    public final String op;

    /**
     * Creates a new Value with no inputs (a leaf node in the graph).
     *
     * @param data The scalar value.
     */
    public Value(double data) {
        this(data, new HashSet<>(), "");
    }

    /**
     * Creates a new Value with inputs (an intermediate node in the graph).
     *
     * @param data     The scalar value.
     * @param children The set of input Values that produced this one.
     * @param op       The operation symbol (e.g. "+", "*") for debugging/display.
     */
    public Value(double data, Set<Value> children, String op) {
        this.data = data;
        this.grad = 0;
        this._backward = () -> {
        };
        this._prev = children;
        this.op = op;
    }

    // --- Operations: each one does two things ---
    // 1. Computes the forward result (normal math)
    // 2. Defines the backward rule (how gradients flow back through this operation)

    /**
     * Addition: d(a+b)/da = 1, d(a+b)/db = 1.
     * Both inputs get the full gradient from the output.
     *
     * @param other The value to add.
     * @return A new Value representing the sum.
     */
    public Value add(Value other) {
        Set<Value> children = new HashSet<>();
        children.add(this);
        children.add(other);

        Value out = new Value(this.data + other.data, children, "+");
        out._backward = () -> {
            this.grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out;
    }

    /**
     * Addition: a + b (double).
     *
     * @param other The value to add.
     * @return A new Value representing the sum.
     */
    public Value add(double other) {
        return this.add(new Value(other));
    }

    /**
     * Multiplication: d(a*b)/da = b, d(a*b)/db = a.
     * Each input's gradient is scaled by the OTHER input's value.
     * Intuition: if b is large, a small change in a has a big effect on the
     * product.
     *
     * @param other The value to multiply by.
     * @return A new Value representing the product.
     */
    public Value mul(Value other) {
        Set<Value> children = new HashSet<>();
        children.add(this);
        children.add(other);

        Value out = new Value(this.data * other.data, children, "*");
        out._backward = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }

    /**
     * Multiplication: a * b (double).
     *
     * @param other The value to multiply by.
     * @return A new Value representing the product.
     */
    public Value mul(double other) {
        return this.mul(new Value(other));
    }

    /**
     * Power: d(x^n)/dx = n * x^(n-1) — the classic calculus power rule.
     *
     * @param exponent The exponent.
     * @return A new Value representing the result.
     */
    public Value pow(double exponent) {
        Set<Value> children = new HashSet<>();
        children.add(this);

        Value out = new Value(Math.pow(this.data, exponent), children, "**" + exponent);
        out._backward = () -> {
            this.grad += exponent * Math.pow(this.data, exponent - 1) * out.grad;
        };
        return out;
    }

    // Other arithmetic operations defined in terms of add, mul, and pow.
    /**
     * Negation: -x.
     *
     * @return A new Value representing the negated value.
     */
    public Value neg() {
        return this.mul(-1);
    }

    /**
     * Subtraction: a - b.
     * Implemented as a + (-b).
     *
     * @param other The value to subtract.
     * @return A new Value representing the difference.
     */
    public Value sub(Value other) {
        return this.add(other.neg());
    }

    /**
     * Subtraction: a - b (double).
     *
     * @param other The value to subtract.
     * @return A new Value representing the difference.
     */
    public Value sub(double other) {
        return this.add(-other);
    }

    /**
     * Division: a / b.
     * Implemented as a * (b^-1).
     *
     * @param other The value to divide by.
     * @return A new Value representing the quotient.
     */
    public Value div(Value other) {
        return this.mul(other.pow(-1));
    }

    /**
     * Division: a / b (double).
     *
     * @param other The value to divide by.
     * @return A new Value representing the quotient.
     */
    public Value div(double other) {
        return this.mul(Math.pow(other, -1));
    }

    /**
     * Log: d(ln(x))/dx = 1/x — used in computing the loss function.
     *
     * @return A new Value representing the natural logarithm.
     */
    public Value log() {
        Set<Value> children = new HashSet<>();
        children.add(this);

        Value out = new Value(Math.log(this.data), children, "log");
        out._backward = () -> {
            this.grad += (1.0 / this.data) * out.grad;
        };
        return out;
    }

    /**
     * Exp: d(e^x)/dx = e^x — the exponential is its own derivative.
     * Used in softmax to convert raw scores into probabilities.
     *
     * @return A new Value representing e^x.
     */
    public Value exp() {
        Set<Value> children = new HashSet<>();
        children.add(this);

        Value out = new Value(Math.exp(this.data), children, "exp");
        out._backward = () -> {
            this.grad += out.data * out.grad;
        };
        return out;
    }

    /**
     * ReLU (Rectified Linear Unit): max(0, x).
     * The simplest activation function. If x &gt; 0, pass it through. If x &lt; 0,
     * output 0.
     * Gradient: 1 if x &gt; 0, 0 if x &lt; 0. Acts like a gate.
     *
     * @return A new Value representing the ReLU activation.
     */
    public Value relu() {
        Set<Value> children = new HashSet<>();
        children.add(this);

        Value out = new Value(this.data < 0 ? 0 : this.data, children, "ReLU");
        out._backward = () -> {
            this.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
        };
        return out;
    }

    /**
     * Sigmoid: 1 / (1 + e^-x).
     * Used for gating mechanisms (output 0 to 1).
     *
     * @return A new Value representing the sigmoid activation.
     */
    public Value sigmoid() {
        Set<Value> children = new HashSet<>();
        children.add(this);

        double val = 1.0 / (1.0 + Math.exp(-this.data));
        Value out = new Value(val, children, "sigmoid");
        out._backward = () -> {
            this.grad += (out.data * (1.0 - out.data)) * out.grad;
        };
        return out;
    }

    /**
     * Backward(): The heart of learning.
     * <p>
     * Step 1: Build a topological ordering of all Value nodes.
     * (If A feeds into B which feeds into C, we need to process C, then B, then A.)
     * </p>
     * <p>
     * Step 2: Walk backward through this ordering, applying the chain rule at each
     * node. Each node's _backward() function pushes gradients to its inputs.
     * </p>
     * <p>
     * After this runs, every Value in the graph knows its gradient — i.e., how much
     * the final error would change if you nudged that number up a tiny bit.
     * </p>
     */
    public void backward() {
        // Topological sort
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();

        // Iterative DFS to avoid stack overflow on deep graphs
        Stack<Value> stack = new Stack<>();
        Stack<Value> path = new Stack<>(); // To track post-order traversal

        // Using a slightly different iterative approach than C# to be safe in Java
        // We need post-order traversal (children before parents)
        buildTopo(this, visited, topo);

        // The loss node gets gradient 1.0 (the starting point of backprop).
        this.grad = 1.0;

        // Propagate backward in reverse topological order
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i)._backward.run();
        }
    }

    /**
     * Helper to build a topological sort of the graph.
     *
     * @param v       Current node.
     * @param visited Set of visited nodes to avoid cycles/revisiting.
     * @param topo    List to append nodes to in topological order.
     */
    private void buildTopo(Value v, Set<Value> visited, List<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v._prev) {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }

    @Override
    public String toString() {
        return "Value(data=" + data + ", grad=" + grad + ")";
    }
}
