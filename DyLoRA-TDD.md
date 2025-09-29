# DyLoRA-MoE: Technical Design Document

## 1. Introduction

### 1.1. Purpose
This document provides a detailed technical design for the Dynamic LoRA-based Mixture-of-Experts (DyLoRA-MoE) system. It outlines the system architecture, component design, data flows, and implementation strategy, reflecting the current state of the Python codebase.

### 1.2. Scope
The scope of this document covers the design of the `DyLoRA_MoE` class and its constituent components: `ExpertManager`, `DynamicHybridRouter`, `NoveltyDetector`, and `SkillLibrary`. It details the mechanisms for model initialization, dynamic expert expansion, and the continual learning lifecycle as implemented.

### 1.3. Definitions, Acronyms, and Abbreviations
- **MoE:** Mixture-of-Experts
- **LoRA:** Low-Rank Adaptation
- **PEFT:** Parameter-Efficient Fine-Tuning
- **OOD:** Out-of-Distribution
- **LLM:** Large Language Model

### 1.4. References
- DyLoRA-MoE: A Dynamic LoRA-based Mixture-of-Experts Architecture for Continual Skill Acquisition (Technical Paper)
- `dylo_moe/` source code directory

### 1.5. Overview
The DyLoRA-MoE system is a continual learning framework designed to incrementally acquire new skills without catastrophic forgetting. It integrates a large, frozen foundation model with a dynamic pool of lightweight LoRA experts. A novelty detection module automates the creation of new experts, and a dynamic router manages the flow of information, adapting its strategy based on skill maturity.

## 2. System Architecture

### 2.1. Architectural Overview
The DyLoRA-MoE architecture is implemented as a single `torch.nn.Module`, the `DyLoRA_MoE` class. It encapsulates a frozen, pre-trained transformer model and orchestrates the interaction between its sub-components.

```mermaid
graph TD
    subgraph DyLoRA_MoE
        A[Input Data] --> B{add_new_skill};
        B -- "is_novel" --> C[ExpertManager: create_expert];
        B -- "is_novel" --> R[Router: add_expert];
        B -- "is_novel" --> S[SkillLibrary: add_skill];
        
        subgraph forward
            D[Input] --> E{Router};
            E -- "routing_weights" --> F[Combine Outputs];
            
            subgraph Foundation Model (PEFT)
                G[Frozen Backbone]
                H[LoRA Adapters]
            end
            
            E1[Expert 1] --> G;
            E2[Expert 2] --> G;
            
            G --> F;
            F --> I[Output Logits];
        end
    end

    C --> H;
```

### 2.2. Component Breakdown
The architecture consists of five primary Python classes.

*   **2.2.1. `DyLoRA_MoE`:** The main orchestration class. It initializes the foundation model, freezes its parameters, and manages the overall forward pass and skill acquisition logic. It holds instances of all other components.
*   **2.2.2. `ExpertManager`:** Manages the lifecycle of LoRA experts. It uses the `peft` library to add new LoRA adapters (`create_expert`) to the foundation model and to switch between active experts (`set_active_expert`). It also handles dynamic rank allocation for new experts.
*   **2.2.3. `SkillLibrary`:** A simple in-memory key-value store that maps a `skill_id` to a representative skill embedding (`torch.Tensor`). This library provides the knowledge base for the novelty detector.
*   **2.2.4. `NoveltyDetector`:** Determines if incoming data represents a new skill. It calculates the cosine similarity between a new skill's embedding and all embeddings stored in the `SkillLibrary`. If the similarity is below a configured threshold, it signals novelty.
*   **2.2.5. `DynamicHybridRouter`:** A trainable gating network that directs input tokens to the appropriate expert(s). It contains a `torch.nn.Linear` layer and logic to switch between dense (softmax) and sparse (top-k) routing based on the maturity state of the experts.

### 2.3. Data Flow Diagram
The data flow illustrates the two primary operations: the forward pass for inference/training and the `add_new_skill` process for model expansion.

```mermaid
sequenceDiagram
    participant Trainer
    participant DyLoRA_MoE
    participant NoveltyDetector
    participant ExpertManager
    participant Router

    par Forward Pass
        Trainer->>DyLoRA_MoE: forward(input_ids)
        DyLoRA_MoE->>Router: get routing_weights
        loop For Each Expert
            DyLoRA_MoE->>ExpertManager: set_active_expert(i)
            DyLoRA_MoE->>DyLoRA_MoE: foundation_model(input_ids)
        end
        DyLoRA_MoE-->>Trainer: Return weighted logits
    and Add New Skill
        Trainer->>DyLoRA_MoE: add_new_skill(skill_data)
        DyLoRA_MoE->>NoveltyDetector: is_novel(embedding)?
        alt Novel Skill
            NoveltyDetector-->>DyLoRA_MoE: return True
            DyLoRA_MoE->>ExpertManager: create_expert()
            DyLoRA_MoE->>Router: add_expert()
        else Not Novel
            NoveltyDetector-->>DyLoRA_MoE: return False
        end
    end
```

## 3. Detailed Design

### 3.1. `DyLoRA_MoE` Class
*   **3.1.1. Initialization (`__init__`):**
    1.  Loads a pre-trained model from Hugging Face using `AutoModelForCausalLM.from_pretrained`.
    2.  Sets `use_cache=False` on the model config to prevent issues with `DynamicCache` objects during evaluation.
    3.  Unties the `lm_head` from the token embeddings if they are tied, creating a new, trainable `lm_head`.
    4.  Instantiates the `ExpertManager` and creates the first default expert.
    5.  Freezes all parameters of the foundation model except for those containing "lora" or "lm_head" in their names.
    6.  Instantiates the `DynamicHybridRouter`, `SkillLibrary`, and `NoveltyDetector`.
*   **3.1.2. Forward Pass (`forward`):**
    *   **Single-Expert Case:** For efficiency, if only one expert exists, it performs a direct forward pass through the foundation model and returns the logits.
    *   **Multi-Expert Case:**
        1.  Obtains the final hidden states from the transformer backbone.
        2.  Passes the hidden states to the `router` to get routing weights.
        3.  Iterates through each expert, setting it as active using `expert_manager.set_active_expert()`.
        4.  For each active expert, it performs a full forward pass to get the expert's logits.
        5.  Combines the expert logits using a weighted sum, where the weights are the mean of the routing weights across the sequence dimension.
        6.  Calculates the cross-entropy loss if `labels` are provided.
        7.  Returns a `CausalLMOutputWithPast` object, ensuring `past_key_values` is `None`.
*   **3.1.3. Skill Acquisition (`add_new_skill`):**
    1.  Generates a skill embedding by passing the `skill_data` through the model and calculating the mean of the normalized final hidden states.
    2.  Calls `novelty_detector.is_novel()` with the embedding.
    3.  If the skill is novel:
        *   Calls `expert_manager.create_expert()`, potentially with an increased rank.
        *   Calls `router.add_expert()` to expand the gating network.
        *   Calls `skill_library.add_skill()` to store the new skill's embedding.
        *   Sets the maturity of the *previous* expert to 1, marking it as mature.

### 3.2. `DynamicHybridRouter` Class
*   **3.2.1. State:**
    *   `gate`: An `nn.Linear` layer that maps the input hidden size to the number of experts.
    *   `expert_maturity`: A `torch.Tensor` of size `(num_experts,)` storing the maturity state (0 for new, 1 for mature) of each expert.
*   **3.2.2. Routing Logic (`forward`):**
    *   If `any(self.expert_maturity == 0)`, it applies a softmax function to the gate logits (divided by a temperature parameter) to produce dense routing weights.
    *   Otherwise (all experts are mature), it performs a `top_k` operation on the logits and applies softmax only to the top-k values, creating sparse weights.
*   **3.2.3. Expansion (`add_expert`):**
    *   Increments `self.num_experts`.
    *   Creates a new `nn.Linear` layer with an output dimension of `self.num_experts`.
    *   Copies the weights and biases from the old gate to the new one.
    *   Appends a `0` to the `expert_maturity` tensor.

### 3.3. `ExpertManager` Class
*   **3.3.1. State:**
    *   `model`: A `Union[PreTrainedModel, PeftModel, PeftMixedModel]` instance that holds the base model and its LoRA adapters.
    *   `expert_configs`: A dictionary mapping `expert_id` to its `LoraConfig`.
    *   `current_expert_id`: An integer tracking the currently active expert on the GPU.
*   **3.3.2. Expert Creation (`create_expert`):**
    *   Dynamically determines the `target_modules` for LoRA (e.g., `["q_proj", "v_proj"]`) by inspecting the model's named modules for common attention layer names. This makes the manager model-agnostic.
    *   Creates a `LoraConfig` with the specified rank, alpha, and dropout.
    *   If the model is not yet a `PeftModel`, it calls `get_peft_model` to wrap it. Otherwise, it calls `model.add_adapter` to add the new expert.
    *   **Memory Optimization:** Immediately after creation, the new expert's adapter layers (`LoraLayer`) are moved to CPU memory using `.to('cpu')`. This ensures that only the active expert consumes GPU VRAM.
*   **3.3.3. Expert Switching (`set_active_expert`):**
    *   **Memory Optimization:** Before activating a new expert, it first identifies the adapter modules of the `current_expert_id` (if any) and moves them to CPU memory.
    *   It then identifies the adapter modules for the new `expert_id` and moves them to the model's active device (e.g., 'cuda').
    *   Finally, it calls `self.model.set_adapter()` with the name of the desired expert adapter (e.g., `"expert_0"`) to activate it for the forward pass.

### 3.4. `NoveltyDetector` and `SkillLibrary`
*   **`SkillLibrary`:** A simple wrapper around a dictionary (`skill_embeddings`) that stores `torch.Tensor` embeddings. `get_all_skills()` stacks these into a single tensor for comparison.
*   **`NoveltyDetector`:**
    *   Its `is_novel()` method retrieves all skill embeddings from the library.
    *   It computes the cosine similarity between the input `data_embedding` and the library embeddings.
    *   It returns `True` if the maximum similarity is less than `self.similarity_threshold`.

## 4. Data Management and Training

### 4.1. Data Flow in `poc_train.py`
1.  **Data Loading:** The script downloads and prepares the CodeAlpaca (Python subset) and MBPP datasets.
2.  **Skill Stream:** A `data_stream` is created, where each item is a list of strings representing a distinct skill (e.g., Python code, `requests` library usage).
3.  **Preprocessing:**
    *   `preprocess_training_dataset`: Tokenizes and optionally packs skill data into a `Dataset` object for training. Sequence packing is used to improve efficiency.
    *   `preprocess_evaluation_dataset`: Tokenizes the evaluation datasets (MBPP and CodeAlpaca).
4.  **Continual Learning Loop:**
    *   The script iterates through the `data_stream`.
    *   For each skill, it calls `model.add_new_skill()`.
    *   If a new expert is created (`is_novel` is True), it initiates a training run using the Hugging Face `Trainer` with the new skill's data as `train_dataset`.
    *   After training, the new expert's maturity is set to 1.
    *   Routing metrics and per-domain evaluation losses are logged to `wandb`.

## 5. Deployment and Operational Considerations

*(This section remains conceptual as it pertains to future deployment.)*

### 5.1. Deployment Strategy
The DyLoRA-MoE system will be deployed as a containerized application. The foundation model and the LoRA experts will be packaged into a single container image. The system will be deployed on a cloud platform that provides GPU acceleration.

### 5.2. Monitoring and Logging
- **Monitoring:** Key metrics to monitor include latency, GPU utilization, number of active experts, and the rate of new skill acquisition.
- **Logging:** The system will log significant events, including the creation of new experts, changes in routing strategy, and errors.

## 6. Future Work and Extensions
- **Expert Pruning:** Develop a mechanism to prune or merge experts that are redundant or no longer in use.
- **Load Balancing Loss:** Introduce an auxiliary loss term to encourage the router to distribute load more evenly across experts, preventing specialization collapse.
- **Automated Hyperparameter Tuning:** Develop a system for automatically tuning the hyperparameters of the LoRA experts and the router.


## 2. System Architecture

### 2.1. Architectural Overview
The DyLoRA-MoE architecture is built upon a large, pre-trained transformer model that remains frozen. New skills are encapsulated in lightweight, dynamically added LoRA experts. A novelty detection module triggers the creation of new experts, and a dynamic router manages the flow of information, adapting its strategy based on skill maturity.

```mermaid
graph TD
    subgraph DyLoRA-MoE System
        A[Input Data] --> B{Novelty Detection};
        B -- "New Skill" --> C[Instantiate New LoRA Expert];
        B -- "Existing Skill" --> D{Dynamic Hybrid Router};
        A --> D;
        D -- "Dense Routing (New Skill)" --> E1[LoRA Expert 1];
        D -- "Dense Routing (New Skill)" --> E2[LoRA Expert 2 ...];
        D -- "Sparse Routing (Mature Skill)" --> EN[LoRA Expert N];
        
        subgraph Frozen Foundation Model
            direction LR
            F[Attention Layers]
            G[FFN Layers]
        end

        E1 --> F;
        E2 --> F;
        EN --> F;

        F --> H[Output];
        G --> H;
    end

    C --> E1;
```

### 2.2. Component Breakdown
The architecture consists of four primary components working in concert.

*   **2.2.1. Frozen Foundation Model Backbone:** A large, pre-trained transformer model (e.g., LLM or VLM) that serves as a stable knowledge reservoir and feature extractor. Its parameters are not updated during skill acquisition.
*   **2.2.2. Dynamic LoRA Experts:** A pool of lightweight, parameter-efficient LoRA modules. Each expert encapsulates a specific skill and is applied to the frozen backbone. The pool can be expanded dynamically.
*   **2.2.3. Novelty-Driven Expansion Trigger:** An automated mechanism that monitors the system's performance and confidence. It triggers the creation of a new LoRA expert when it encounters out-of-distribution data that existing experts cannot handle effectively.
*   **2.2.4. Dynamic Hybrid Router:** A trainable gating network that directs input tokens to the appropriate expert(s). It dynamically shifts from a dense, collaborative routing strategy for new skills to a sparse, delegation-based strategy for mature skills.

### 2.3. Data Flow Diagram
The data flow illustrates the lifecycle of a request through the system, from initial processing to final output.

```mermaid
sequenceDiagram
    participant User
    participant System
    participant NoveltyDetector as Novelty Detection
    participant ExpertManager as LoRA Expert Manager
    participant Router as Dynamic Hybrid Router
    participant FoundationModel as Frozen Backbone

    User->>System: Process Input Data
    System->>NoveltyDetector: Analyze Data
    alt New Skill Detected
        NoveltyDetector->>System: Signal Novelty
        System->>ExpertManager: Instantiate New Expert
        ExpertManager-->>System: New Expert Ready
    end
    System->>Router: Route Input
    Router->>FoundationModel: Process with relevant Expert(s)
    FoundationModel-->>System: Generate Output
    System-->>User: Return Result
```

## 3. Detailed Design

### 3.1. Frozen Foundation Model Backbone
*   **3.1.1. Model Selection Criteria:** The foundation model should be a large, pre-trained transformer model with strong performance on a wide range of tasks. The choice of model will depend on the target domain (e.g., Llama 3 for language, a VLM for multimodal tasks). The model must be compatible with the Hugging Face `transformers` and `peft` libraries.
*   **3.1.2. Implementation Details:** The model will be loaded using the `transformers` library, and its parameters will be frozen by setting `requires_grad=False` for all layers.

### 3.2. Dynamic LoRA Experts
*   **3.2.1. LoRA Configuration:** Each LoRA expert will consist of a set of low-rank matrices applied to the attention and/or FFN layers of the backbone. The rank (`r`) and scaling factor (`alpha`) of the LoRA matrices will be configurable, allowing for a trade-off between parameter efficiency and expressive power.
*   **3.2.2. Expert Management Service:** A dedicated service will be responsible for creating, storing, and retrieving LoRA experts. This service will maintain a registry of all active experts and their associated skills.

### 3.3. Novelty-Driven Expansion Trigger
*   **3.3.1. Triggering Mechanisms:** The novelty detection mechanism will be implemented using one or both of the following strategies:
    *   **Router Confidence Monitoring:** The system will monitor the entropy of the router's output distribution. High entropy will indicate that no single expert is confident, suggesting a new skill.
    *   **Gradient-Based OOD Detection:** The Mahalanobis distance of gradients will be used to identify inputs that are statistically dissimilar to the training data of existing experts.
*   **3.3.2. API Definition:** The novelty trigger will expose an API to the system that allows it to check if a given input is novel.

### 3.4. Dynamic Hybrid Router
*   **3.4.1. Routing Logic:** The router will be a small neural network that takes token embeddings as input and outputs a probability distribution over the available experts. The routing logic will be implemented as follows:
    *   **Dense Collaboration:** For new skills, the router will use a "soft" routing policy, computing a weighted average of the outputs from the new expert and one or more relevant established experts.
    *   **Sparse Delegation:** For mature skills, the router will transition to a sparse top-k routing policy (e.g., top-1), delegating the task to the most specialized expert.
*   **3.4.2. Training Strategy:** The router will be trained in parallel with the LoRA experts. Its training objective will be to minimize the task loss while gradually shifting from a dense to a sparse routing policy as the experts mature.
*   **3.4.3. State Management:** The router will maintain a state for each expert, tracking its maturity level. This state will be used to determine the appropriate routing strategy.

## 4. Skill Acquisition Lifecycle

### 4.1. Phase 1: Novelty Detection
The system continuously monitors incoming data using the novelty detection module. When an out-of-distribution input is detected, the system flags it as a new skill.

### 4.2. Phase 2: LoRA Expert Instantiation
A new, randomly initialized LoRA expert is created and added to the expert pool. The expert is associated with the new skill.

### 4.3. Phase 3: Few-Shot, High-Rate Seeding
The new LoRA expert is trained on the few available examples of the new skill using a high learning rate. This allows the expert to quickly learn the basic features of the new skill.

### 4.4. Phase 4: Low-Rate Consolidation
As more data for the new skill becomes available, the expert is further trained using a low, stable learning rate. This allows the expert to fine-tune its parameters and converge to a robust solution.

### 4.5. Phase 5: Router Adaptation and Specialization
The router is trained in parallel with the expert. Initially, it learns to use the dense collaboration strategy. As the expert matures, the router's training objective shifts to favor the sparse delegation policy.

## 5. Data Management

### 5.1. Data Storage
- **Skill Data:** The few-shot examples for each new skill will be stored in a dedicated database or file system. This data will be versioned and associated with the corresponding LoRA expert.
- **Expert Registry:** The Expert Management Service will maintain a registry of all LoRA experts. This registry will be stored in a database and will contain metadata about each expert, including its skill, version, and training history.

### 5.2. Data Schemas
- **Skill Data Schema:**
  - `skill_id`: Unique identifier for the skill.
  - `expert_id`: Identifier of the LoRA expert associated with the skill.
  - `input_data`: The input data for the skill.
  - `output_data`: The expected output for the skill.
- **Expert Registry Schema:**
  - `expert_id`: Unique identifier for the expert.
  - `skill_id`: Identifier of the skill the expert is trained on.
  - `lora_config`: The configuration of the LoRA matrices (r, alpha, etc.).
  - `storage_path`: The path to the stored expert weights.
  - `maturity_state`: The current maturity level of the expert.

## 6. Deployment and Operational Considerations

### 6.1. Deployment Strategy
The DyLoRA-MoE system will be deployed as a containerized application. The foundation model and the LoRA experts will be packaged into a single container image. The system will be deployed on a cloud platform that provides GPU acceleration.

### 6.2. Monitoring and Logging
- **Monitoring:** The system will be monitored for performance and resource utilization. Key metrics to monitor include:
  - Latency of predictions.
  - GPU utilization.
  - Number of active experts.
  - Rate of new skill acquisition.
- **Logging:** The system will log all significant events, including:
  - Creation of new experts.
  - Changes in routing strategy.
  - Errors and exceptions.

### 6.3. Scalability and Performance
The DyLoRA-MoE architecture is designed to be scalable. The use of LoRA experts ensures that the model size grows slowly as new skills are added. The dynamic routing mechanism allows the system to adapt its computational cost based on the maturity of the skills.

## 7. Future Work and Extensions
- **Expert Pruning:** Develop a mechanism to prune or merge experts that are redundant or no longer in use.
- **Multi-Task Learning:** Extend the framework to support multi-task learning, where a single input can be routed to multiple experts simultaneously.
- **Automated Hyperparameter Tuning:** Develop a system for automatically tuning the hyperparameters of the LoRA experts and the router.