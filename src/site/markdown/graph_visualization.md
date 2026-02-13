# The MicroGPT Dynamic Computation Graph

This document visualizes how `MicroGPT` builds its computation graph on the fly ("Define-by-Run").

## Legend
- **Orange/Red Rectangles**: Permanent Weights (The "Brain"). These persist throughout the entire life of the program.
- **Pink/Yellow Ovals**: Temporary Values (The "Thoughts"). These are created for a single token and destroyed immediately after (or cached).

## 1. The Components

```mermaid
graph TD
    subgraph Permanent_Memory ["PERMANENT MEMORY (Weights)"]
        direction TB
        WTE["Token Embeddings (wte)"]
        WPE["Pos Embeddings (wpe)"]
        WQ["Query Weights (attn_wq)"]
        WK["Key Weights (attn_wk)"]
        WV["Value Weights (attn_wv)"]
    end

    subgraph Temporary_Graph ["TEMPORARY GRAPH (Per Token)"]
        direction TB
        x_in["Input x"]
        q_vec["Query Vector"]
        k_vec["Key Vector"]
        v_vec["Value Vector"]
        logits["Logits (Output)"]
    end

    %% High Contrast Orange for Dark Mode
    style WTE fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style WPE fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style WQ fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style WK fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style WV fill:#ff9900,stroke:#333,stroke-width:2px,color:#000

    style x_in fill:#f9f,stroke:#333,color:#000
    style q_vec fill:#f9f,stroke:#333,color:#000
    style k_vec fill:#f9f,stroke:#333,color:#000
    style v_vec fill:#f9f,stroke:#333,color:#000
    style logits fill:#f9f,stroke:#333,color:#000
```

## 2. Construction (Single Token Pass)
When `gpt()` runs for a token, it temporarily links the **Permanent Weights** to new **Temporary Values**.

```mermaid
graph TD
    %% Permanent Weights
    WTE["Weights: Token Emb"]
    WQ["Weights: Query"]

    %% Temporary Values
    t_emb["Temp: 'a' vector"]
    q_vec["Temp: Query Vector"]
    
    %% The Connections
    WTE -->|Lookup| t_emb
    t_emb -->|Input| q_vec
    WQ -->|Multiply| q_vec
    
    %% Styling
    style WTE fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    style WQ fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    
    style t_emb fill:#f9f,stroke:#333,color:#000
    style q_vec fill:#f9f,stroke:#333,color:#000
```

## 3. Training vs Inference Lifecycle

### Training (Forward + Backward)
1.  **Build**: We build the full graph for a sequence (e.g., 32 tokens).
2.  **Hold**: We keep *everything* in memory.
3.  **Backward**: We traverse the graph from `Loss` back to `Weights`, calculating gradients.
4.  **Update**: We modify the `Weights` (Orange Nodes).
5.  **Destroy**: We delete all Temporary/Pink Nodes.

### Inference (Generation)
1.  **Build**: We build the graph for *one* token.
2.  **Predict**: We get the next token.
3.  **Cache**: We save *only* the Key/Value vectors (Yellow) to the KV Cache.
4.  **Destroy**: We delete everything else (q, logits, hidden states) immediately.

```mermaid
graph TD
    subgraph KV_Cache ["KV CACHE (Preserved)"]
        K_history["Past Keys"]
        V_history["Past Values"]
    end
    
    subgraph Trash_Bin ["GARBAGE COLLECTED"]
        q_vec["Query Vector"]
        x_vec["Hidden States"]
        logits["Logits"]
    end
    
    style K_history fill:#ff9,stroke:#333,color:#000
    style V_history fill:#ff9,stroke:#333,color:#000
    style q_vec fill:#f9f,stroke:#333,color:#000
    style x_vec fill:#f9f,stroke:#333,color:#000
    style logits fill:#f9f,stroke:#333,color:#000
```

## 4. The Training Process (Step-by-Step)
This diagram illustrates how the optimizer interacts with the weights.

```mermaid
graph TD
    subgraph Weights ["1. PERMANENT MEMORY"]
        direction TB
        W_in["Weights"]
    end

    subgraph Forward ["2. FORWARD PASS"]
        direction TB
        Input[("Input Data")] -->|Combine| Graph["Dynamic Graph"]
        W_in -->|Used By| Graph
        Graph -->|Produces| Output["Prediction"]
    end

    subgraph Loss_Calc ["3. LOSS CALCULATION"]
        direction TB
        Output -->|Compare| Loss["Loss Value"]
        Target[("Target Data")] -->|Compare| Loss
    end

    subgraph Backward ["4. BACKWARD PASS"]
        direction TB
        Loss -->|Calculate Grads| Gradients["Gradients (.grad)"]
    end

    subgraph Optimizer ["5. OPTIMIZATION"]
        direction TB
        Gradients -->|Input| Adam["Adam Optimizer"]
        Adam -->|Update| W_in
    end

    %% High Contrast Orange for Weights
    style W_in fill:#ff9900,stroke:#333,stroke-width:2px,color:#000
    
    %% Pink for Dynamic Graph
    style Graph fill:#f9f,stroke:#333,color:#000
    style Output fill:#f9f,stroke:#333,color:#000
    style Loss fill:#f9f,stroke:#333,color:#000
    
    %% Blue for Optimizer
    style Adam fill:#add8e6,stroke:#333,color:#000
    style Gradients fill:#add8e6,stroke:#333,color:#000
```
