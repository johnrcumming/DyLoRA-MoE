## DyLoRA-MoE POC Training Run Analysis & Improvement Plan (2025-08-27)

### 1. Executive Summary
POC trained Gemma3-270M with dynamic LoRA experts. Only one new expert (expert_1) was added; subsequent skills were classified as non-novel and skipped. Validation (MBPP) loss improved marginally: 1.5133 → 1.5055 (≈0.5% relative). Training consumed 3,227s for 300 steps (0.232 steps/s) on Apple MPS with gradient checkpointing and very small effective batch (2 * grad_accum 2 = 4 tokens batches of length ≤512). Grad norms logged are near-zero or extremely small, suggesting frozen base + very low signal to LoRA layers and/or logging mismatch. Router stayed at 2 experts; routing maturity logic kept dense softmax routing (no sparsification).

### 2. Key Observations
| Area | Observation | Impact |
|------|-------------|--------|
| Novelty detection | High similarity threshold (0.85) + crude per-skill embedding (mean last-layer hidden states of limited samples) likely overestimates similarity. | Prevents creation of additional experts → under-utilizes MoE design. |
| Grad norms | Many near-zero grad_norm entries (e.g., 5e-09) with occasional spikes; potential cause: manual loss vs Trainer expectation, adapter freezing, or FP16/precision mismatch. | Risk of under-training LoRA adapters; misleading optimization diagnostics. |
| Routing | For 2 experts and maturity vector containing a 0 (new expert marked mature only after training), dense softmax used throughout first expert's training; after maturity set to 1 for new expert 1, expert 0 may remain 0 (never explicitly matured). | Always dense routing, undermining sparsity/ specialization benefits. |
| Evaluation set | Uses Code Alpaca test for eval during training while novelty detection & final metric rely on MBPP. Domain shift between training skill stream and eval metric may blur learning gains. | Small MBPP gains may not reflect skill acquisition quality. |
| Data stream | Python code (1000 outputs) tokenized as raw completions; other “skills” are tiny (2 examples each). | Insufficient data per subsequent skill to justify novelty detection; statistical noise. |
| Learning schedule | Cosine LR with warmup, but training ended early (1.2 epochs) due to early stopping? (eval every 50 steps, patience=5; plateau at eval_loss 1.8547 indicates using Code Alpaca eval metric, not MBPP) | Early stopping prevented full planned epochs. |
| Loss plateau | Eval loss stuck at ~1.8547 on Code Alpaca test while train loss fluctuates 1.44–2.31; weak generalization improvements. | Need better regularization or curriculum & adapter capacity. |
| Trainable % | 0.0847% (≈370K params) of 436M. | Possibly too low for multiple heterogeneous skills; increase r for specific experts. |
| Tokenization | Fixed max_length=512 with padding to max_length in evaluation; training dataset constructed with padding=True (dynamic) but not packing multiple short samples. | Wasted compute due to padding tokens; lower effective tokens/s. |
| Logging | total_flos = 0 in W&B summary. | FLOPs accounting not propagated; indicates Trainer hooks not receiving model outputs in expected format due to custom forward signature returning (loss, output). |

### 3. Root Cause Analysis of Limited Improvement
1. Conservative novelty detection prevented creation of specialized experts beyond the first new one. 
2. Evaluation signal mismatch (Code Alpaca vs MBPP) guided early stopping toward a metric weakly correlated with targeted skill improvements.
3. Potential gradient underflow / logging issue reduced effective optimization of LoRA weights.
4. Under-capacity adapters (r=8, alpha=16) may constrain representational shift needed for new APIs (requests, stripe, flask) vs base model.
5. Inefficient token utilization (padding to 512, tiny batch, no packing) reduces gradient quality per wall-clock second.
6. Router never transitions to sparse routing due to maturity flags, limiting specialization pressure.

### 4. Recommended Immediate Fixes (Low Risk)
1. Novelty Detection:
   - Lower similarity_threshold to 0.75 initially; optionally use median similarity vs max.
   - Use pooled embedding from multiple random batches (e.g., 32 samples) and mean after LayerNorm.
2. Expert Creation Policy:
   - Add a minimum sample count per skill before decision (e.g., require ≥32 snippets).
   - Add temperature decay in router after expert matures.
3. Maturity Handling:
   - Mark initial expert mature at initialization; set both experts’ maturity to 1 after N warmup steps for new expert, enabling sparse routing.
4. Gradient Logging / FLOPs:
   - Adjust model.forward to return a CausalLMOutput so Trainer can compute flops and grad norm reliably (currently returns (loss, output)).
5. Data Efficiency:
   - Implement sequence packing (concatenate multiple short examples until 512 tokens) to reduce padding fraction.
   - Increase per_device_train_batch_size if memory allows (MPS: test 4) and reduce grad_accumulation.
6. Evaluation Consistency:
   - Add MBPP eval to training loop (multi-metric) and early stop on MBPP loss if that’s target.
7. Adapter Capacity:
   - For API skills, set higher r (e.g., r=16) for new expert only; maintain r=8 for base expert—requires per-expert config support.

### 5. Medium-Term Enhancements
1. Adaptive Similarity Threshold: decay threshold as library grows: tau_k = base - delta * log(1 + k).
2. Skill Embedding Model: Replace raw hidden-state averaging with projection head (MLP + normalization) trained contrastively (SimCLR-style) on skill vs random mix to sharpen distance metric.
3. Router Improvements:
   - Add load balancing / auxiliary entropy loss to encourage specialization.
   - Implement top-2 routing with capacity factor and combine logits weighted by normalized gate scores.
4. Expert Freezing Schedule: After expert converges (no improvement N evals), partially freeze its LoRA layers (e.g., q_proj only) to protect knowledge.
5. Dynamic Adapter Rank: Start new expert with small r and grow (r doubling) if validation improvement < epsilon after M steps.
6. Knowledge Distillation Backward: Regularize new expert outputs toward ensemble of mature experts to prevent divergence.
7. Token-Level Novelty: Detect novelty per sample; batch similar novel samples for focused micro-training bursts.

### 6. Longer-Term Research Directions
1. Meta-Router: Learn a meta-policy to predict when to spawn experts (reinforcement learning over downstream eval gain).
2. Adapter Merging: Periodically cluster low-usage experts and merge via weight averaging + small finetune to control expert explosion.
3. Continual Replay Buffer: Store exemplars per skill to perform rehearsal when training new expert, mitigating interference in shared router / lm_head.
4. Parameter Scaling Law: Empirically derive performance vs adapter rank curve to inform automated rank scheduling.
5. Cross-Skill Composition: Allow multiple experts active per token with learned soft combination (Mixture of LoRA sums) vs gating final logits only.

### 7. Quantitative Targets for Next Iteration
| Metric | Current | Target (Next POC) | Stretch |
|--------|---------|------------------|---------|
| MBPP validation loss delta | -0.0078 | -0.03 | -0.05 |
| Additional experts created | 1 | 2–3 | 4 |
| Steps/sec | 0.232 | 0.35 | 0.45 |
| Trainable params (%) | 0.085% | 0.12% | 0.18% |
| Padding ratio | ~>40% (est.) | <20% | <10% |

### 8. Implementation Plan (Actionable) – Status
Legend: [x] done, [~] partial, [ ] pending

1. Refactor forward: [x]
   - CausalLMOutputWithPast now returned directly.
2. Novelty Detector Adjustments: [~]
   - Lowered threshold to 0.75; median option added. (Moving average & improved pooling still pending.)
3. Skill Embedding Extraction: [ ]
   - Still using simple mean of hidden states (improvement pending).
4. Router Maturity & Sparse Switch: [~]
   - Initial expert marked mature; new expert maturity set post-train. (Warmup-based dual maturation & sparse top-k transition logic not yet added.)
5. Add MBPP to eval loop: [~]
   - Combined eval dataset includes MBPP + Python; no per-domain metric separation / early stop on MBPP yet.
6. Sequence Packing Utility: [x]
   - Naive packing added; padding_fraction logged.
7. Per-Expert Rank: [x]
   - create_expert accepts r/alpha overrides; dynamic r doubling logic added for new experts.
8. Routing Metrics Logging: [x]
   - routing_entropy & expert_usage_* logged.
9. Padding & Efficiency Metrics: [x]
   - padding_fraction logged; tokens_per_second still pending.
10. FLOPs / Grad Norm Visibility: [x]
   - Forward signature aligned with Trainer expectations (should restore FLOPs; verify next run).

### 8a. Immediate Next Steps (Short List)
1. Implement per-domain evaluation metrics & custom compute_metrics to log mbpp_eval_loss separately.
2. Introduce warmup-based maturity schedule + switch to top-k routing once all experts mature.
3. Add improved skill embedding pooling (e.g., last hidden state mean after RMSNorm) & moving average similarity tracker.
4. Log tokens_per_second (compute from batch_size * seq_len / runtime between steps).
5. Optional: temperature decay for router after expert maturation.

### 9. Risk & Mitigation
| Risk | Mitigation |
|------|------------|
| Over-spawning experts | Cooldown (min steps between creations) & performance delta threshold. |
| Memory increase from larger r | Monitor peak memory; cap total trainable params; evict least-used experts. |
| Router instability after sparse switch | Gradually anneal temperature; start with top-k=2 then 1. |
| Noisy novelty from small sample | Require minimum token count & similarity confidence interval. |

### 10. Minimal Code Changes (Next PR Checklist – Updated)
[x] Modify `DyLoRA_MoE.forward` output
[x] Alter initial expert maturity setup (initial expert marked mature)
[~] Add dynamic threshold & embedding refactor in NoveltyDetector / SkillLibrary (threshold + median added; advanced embedding pending)
[x] Add sequence packing function in `poc_train.py`
[x] Introduce per-expert r override
[~] Add MBPP eval each eval_steps (combined dataset present; per-domain metrics & early stopping pending)
[x] Log routing_entropy & expert_usage
[x] Log padding_fraction
[ ] Log tokens_per_second
[ ] Implement sparse top-k routing transition

### 13. Change Log (2025-08-27)
- Forward method now returns standard CausalLMOutputWithPast.
- NoveltyDetector threshold lowered to 0.75 with optional median heuristic.
- Per-expert LoRA rank overrides enabled; dynamic r doubling for new experts (cap 32).
- Initial expert marked mature; new expert maturity set post-training.
- Sequence packing implemented; padding_fraction logged.
- Combined eval dataset (Python + MBPP) created for Trainer; final separate MBPP test still executed.
- Added logging of routing_entropy and expert_usage_*.
- Added logging of padding_fraction per skill ingestion.
- Added dynamic expert capacity scaling logic and partial routing maturity handling.

Pending (not yet implemented in code): per-domain metrics separation, moving average similarity, tokens_per_second logging, sparse top-k switch, temperature decay, improved embedding pooling, advanced novelty confidence, and router load-balancing losses.

### 11. Monitoring Additions
Log the following to W&B:
 - similarity_scores_on_creation
 - routing_entropy
 - expert_usage (average gate weight per expert)
 - padding_fraction
 - tokens_per_second
 - packed_sequence_length_histogram

### 12. Conclusion
The core DyLoRA-MoE pipeline runs end-to-end and demonstrates dynamic adapter creation, but novelty detection, routing specialization, and data efficiency limit realized gains. The outlined low-risk adjustments should yield clearer expert differentiation and measurable performance improvements in the next iteration.

---
Prepared by: GitHub Copilot (automated analysis)