"""Test different routing strategies."""

import sys
import os

workspace_root = r"w:\UnderTheRadar\DyLoRA-MoE"
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "evalplus"))

from evalplus.provider.peft_moe import PeftMoEDecoder

model_path = "./artifacts/best-dylora-model-full-v16/best_model"
test_prompt = "def add(a, b):\n    return"

for strategy in ["router", "single:0", "single:1", "single:2", "single:3"]:
    print(f"\n{'='*60}")
    print(f"Testing strategy: {strategy}")
    print(f"{'='*60}")
    
    decoder = PeftMoEDecoder(
        name=model_path,
        dataset="humaneval",
        batch_size=1,
        temperature=0.0,
        dtype="bfloat16",
        routing_strategy=strategy,
        trust_remote_code=True,
        max_new_tokens=30
    )
    
    result = decoder.codegen(test_prompt, do_sample=False, num_samples=1)
    print(f"Result: {repr(result[0][:100])}")
