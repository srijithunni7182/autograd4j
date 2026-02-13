# AutoGrad-Java

A complete GPT language model — training and inference — in pure Java with zero dependencies.

Faithful port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) and [Milan's C# AutoGrad-Engine](https://github.com/milanm/AutoGrad-Engine).

## What is this?

This is the exact same algorithm that powers ChatGPT, stripped to its essence. No PyTorch, no TensorFlow, no external libraries. Just plain Java and math.

It trains a tiny GPT model on a list of human names, then generates new ones.

**New to ML?** Start with the [Prerequisites guide](PREREQUISITES.md) — it covers all the math and ML concepts you need, from scratch.

## Project Structure

| File | Responsibility |
|---|---|
| `Value.java` | Autograd engine — wraps scalars with automatic gradient tracking |
| `Tokenizer.java` | Character-level tokenizer with `encode()`/`decode()` |
| `NeuralOps.java` | Stateless neural-net building blocks: `linear`, `softmax`, `rmsNorm` |
| `MicroGPT.java` | Main entry point: model definition, training loop, and generation |

## Quick Start

### Prerequisites
- Java 11 or higher (Tested with Java 21)

### compiling and Running

1. **Compile the project:**
   ```bash
   mkdir -p bin
   javac -d bin src/main/java/com/autograd/*.java
   ```

2. **Run the training and generation:**
   ```bash
   java -cp bin com.autograd.MicroGPT
   ```

3. **Run with custom hyperparameters:**
   ```bash
   java -cp bin com.autograd.MicroGPT --n_embd 32 --n_layer 2 --num_steps 2000
   ```

### Running Tests

To verify the autograd engine and model components:

1. **Compile tests:**
   ```bash
   javac -d bin src/main/java/com/autograd/*.java src/test/java/com/autograd/*.java
   ```

2. **Run the test runner:**
   ```bash
   java -cp bin com.autograd.TestRunner
   ```


## Documentation

Full project documentation, including Javadocs and the Visualization Guide, is available in the `docs` directory.

- [Open Documentation](docs/index.html) (Open in your browser)

## License

MIT — learn from it, play with it, share it.