import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import time
import evaluate # Added for accuracy metrics

# --- IMPORTANT: Hugging Face Authentication ---
# To use Llama 2, you need to be authenticated.
# 1. Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and accept the terms.
# 2. In your terminal, run: `huggingface-cli login` and paste your access token.


# --- 1. Model and Tokenizer Initialization ---
print("Loading models... This may take a moment, especially for the 7B model.")

# Check for CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("This script requires CUDA-enabled GPUs to run larger models.")

# We use bfloat16 for performance on A100 GPUs
dtype = torch.bfloat16

# --- MODEL SWAP ---
# We are changing the small model to TinyLlama, which is in the same family as Llama 2.
# This ensures architectural and tokenizer compatibility, leading to better results.
small_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Note: For multi-GPU, we explicitly place the small model on the first GPU.
small_model = AutoModelForCausalLM.from_pretrained(
    small_model_name, 
    torch_dtype=dtype
).to("cuda:0")

# Load the larger, more powerful model ("expert").
# device_map="auto" will spread this model across all available GPUs (your 2xA100s).
large_model_name = "meta-llama/Llama-2-7b-chat-hf"
large_model = AutoModelForCausalLM.from_pretrained(
    large_model_name,
    torch_dtype=dtype,
    device_map="auto"  # This is the key for multi-GPU inference
)
# The Llama 2 tokenizer is compatible and preferred for both models.
large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)

# --- Tokenizer Unification ---
# To avoid errors, we'll use the more capable (larger) tokenizer for all operations.
if large_tokenizer.pad_token is None:
    large_tokenizer.pad_token = large_tokenizer.eos_token
    # Also configure the models to use this pad token id
    small_model.config.pad_token_id = large_tokenizer.pad_token_id
    large_model.config.pad_token_id = large_tokenizer.pad_token_id

# We will now use the large_tokenizer for all encoding and decoding.
tokenizer = large_tokenizer

print("Models loaded successfully.")
print(f"Small model ({small_model_name}) is on: {small_model.device}")
print(f"Large model ({large_model_name}) device map: {large_model.hf_device_map}")


# --- 2. Entropy Calculation Function ---
def calculate_entropy_from_logits(logits):
    """Calculates entropy for a batch of logits."""
    probabilities = F.softmax(logits, dim=-1)
    epsilon = 1e-9
    entropy_batch = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=-1)
    return entropy_batch

# --- 3. Threshold Calibration Function ---
def calculate_and_set_entropy_threshold(
    model, 
    tokenizer, 
    dataset_name="wikitext", 
    dataset_config="wikitext-2-raw-v1",
    split="test",
    num_samples=100,
    percentile=80,
    chunk_size=512 
):
    print(f"\n--- Calibrating entropy threshold from '{dataset_name}' ---")
    print(f"Target percentile: {percentile}% (will trigger large model on the top {100-percentile}% of tokens)")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    entropies = []
    print("Processing dataset samples to gather entropy distribution...")
    for i in tqdm(range(min(num_samples, len(dataset)))):
        text = dataset[i]['text']
        if not text:
            continue
        input_ids = tokenizer.encode(text, return_tensors="pt")
        for chunk_start in range(0, input_ids.shape[1], chunk_size):
            chunk_end = min(chunk_start + chunk_size, input_ids.shape[1])
            input_chunk = input_ids[:, chunk_start:chunk_end].to(model.device)
            if input_chunk.shape[1] <= 1:
                continue
            with torch.no_grad():
                outputs = model(input_ids=input_chunk)
                logits = outputs.logits[:, :-1, :] 
                logits_flat = logits.view(-1, logits.size(-1))
                entropy_batch = calculate_entropy_from_logits(logits_flat)
                entropies.extend(entropy_batch.to(torch.float32).cpu().numpy())
    if not entropies:
        raise ValueError("Could not calculate any entropies.")
    threshold = np.percentile(entropies, percentile)
    print(f"Calibration complete. Calculated Threshold at {percentile}th percentile: {threshold:.4f}")
    return threshold


# --- 4. Generation Functions ---

def generate_with_entropy_gate(prompt, max_new_tokens, entropy_threshold):
    """Generates text using the entropy gate and collects performance metrics."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(small_model.device)
    generated_sequence = input_ids.clone()
    
    large_model_calls = 0
    generation_start_time = time.time()

    for i in range(max_new_tokens):
        with torch.no_grad():
            small_model_outputs = small_model(input_ids=generated_sequence.to(small_model.device))
            next_token_logits_small = small_model_outputs.logits[:, -1, :]
            entropy = calculate_entropy_from_logits(next_token_logits_small).item()
            if entropy < entropy_threshold:
                next_token_logits = next_token_logits_small
            else:
                large_model_calls += 1
                large_model_outputs = large_model(input_ids=generated_sequence)
                next_token_logits = large_model_outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_sequence = torch.cat([generated_sequence, next_token_id.to(generated_sequence.device)], dim=-1)

    torch.cuda.synchronize()
    total_time = time.time() - generation_start_time
    final_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    generated_summary = final_text.split("[/INST]")[-1].strip()
    large_model_percentage = (large_model_calls / max_new_tokens) * 100
    return {"time": total_time, "text": generated_summary, "large_model_percentage": large_model_percentage}

def generate_large_only_optimized(prompt, max_new_tokens):
    """FAIR BASELINE: Generates with the large model using its optimized .generate() method (with KV caching)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(large_model.device)
    torch.cuda.synchronize()
    start_time = time.time()
    output_sequence = large_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    final_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    generated_summary = final_text.split("[/INST]")[-1].strip()
    return {"time": total_time, "text": generated_summary}

def generate_small_only_optimized(prompt, max_new_tokens):
    """NEW BASELINE: Generates with the small model using its optimized .generate() method."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(small_model.device)
    torch.cuda.synchronize()
    start_time = time.time()
    output_sequence = small_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    final_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    generated_summary = final_text.split("[/INST]")[-1].strip()
    return {"time": total_time, "text": generated_summary}

def generate_large_only_naive(prompt, max_new_tokens):
    """NAIVE BASELINE: Simulates large model generation without KV caching."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(large_model.device)
    generated_sequence = input_ids.clone()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = large_model(input_ids=generated_sequence)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_sequence = torch.cat([generated_sequence, next_token_id], dim=-1)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    final_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    generated_summary = final_text.split("[/INST]")[-1].strip()
    return {"time": total_time, "text": generated_summary}


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # --- Step 1: Calibrate Threshold ---
    calibrated_threshold = calculate_and_set_entropy_threshold(
        model=large_model, tokenizer=tokenizer, percentile=50
    )
    
    # --- Step 2: Evaluate on a Sample Dataset ---
    print("\n\n=========================================================")
    print("====== Evaluating Gated Generation on Test Prompts ======")
    print("=========================================================")
    
    rouge = evaluate.load('rouge')
    eval_dataset = load_dataset("cnn_dailymail", '3.0.0', split='test', streaming=True)
    all_metrics = []
    
    num_prompts = 5
    for i, example in enumerate(eval_dataset.take(5)):
        prompt = f"[INST] Summarize this article: {example['article']} [/INST]"
        reference = example['highlights']
        
        print(f"\n--- Processing Prompt {i+1}/{num_prompts} ---")
        
        # A) Gated Method
        gated_results = generate_with_entropy_gate(prompt, max_new_tokens=50, entropy_threshold=calibrated_threshold)
        
        # B) Fair Baseline (Large Only, Optimized)
        large_opt_results = generate_large_only_optimized(prompt, max_new_tokens=50)

        # C) Small Model Only Baseline
        small_opt_results = generate_small_only_optimized(prompt, max_new_tokens=50)

        # D) Naive Baseline (Large Only, No KV Cache)
        large_naive_results = generate_large_only_naive(prompt, max_new_tokens=50)

        # Calculate ROUGE for all generations
        gated_rouge = rouge.compute(predictions=[gated_results['text']], references=[reference])
        large_opt_rouge = rouge.compute(predictions=[large_opt_results['text']], references=[reference])
        small_opt_rouge = rouge.compute(predictions=[small_opt_results['text']], references=[reference])
        
        all_metrics.append({
            "gated_time": gated_results['time'], "gated_rougeL": gated_rouge['rougeL'],
            "large_opt_time": large_opt_results['time'], "large_opt_rougeL": large_opt_rouge['rougeL'],
            "small_opt_time": small_opt_results['time'], "small_opt_rougeL": small_opt_rouge['rougeL'],
            "large_naive_time": large_naive_results['time'],
            "large_model_percentage": gated_results['large_model_percentage']
        })
        
        print(f"Gated Time: {gated_results['time']:.2f}s | Large Model Time: {large_opt_results['time']:.2f}s | Small Model Time: {small_opt_results['time']:.2f}s")
        print(f"Gated ROUGE-L: {gated_rouge['rougeL']:.4f} | Large ROUGE-L: {large_opt_rouge['rougeL']:.4f} | Small ROUGE-L: {small_opt_rouge['rougeL']:.4f}")

    # --- Step 3: Aggregate and Display Final Results ---
    avg_gated_time = np.mean([m['gated_time'] for m in all_metrics])
    avg_large_opt_time = np.mean([m['large_opt_time'] for m in all_metrics])
    avg_small_opt_time = np.mean([m['small_opt_time'] for m in all_metrics])
    avg_large_naive_time = np.mean([m['large_naive_time'] for m in all_metrics])
    
    avg_gated_rougeL = np.mean([m['gated_rougeL'] for m in all_metrics])
    avg_large_opt_rougeL = np.mean([m['large_opt_rougeL'] for m in all_metrics])
    avg_small_opt_rougeL = np.mean([m['small_opt_rougeL'] for m in all_metrics])

    avg_large_model_usage = np.mean([m['large_model_percentage'] for m in all_metrics])
    
    # Calculate speedups
    fair_speedup = avg_large_opt_time / avg_gated_time
    naive_speedup = avg_large_naive_time / avg_gated_time
    
    print("\n\n=========================================================")
    print("============== AGGREGATE PERFORMANCE METRICS ==============")
    print("=========================================================")
    print(f"Gated Method (TinyLlama + Llama-7B):")
    print(f"  - Average Time: {avg_gated_time:.2f}s")
    print(f"  - Average ROUGE-L Score: {avg_gated_rougeL:.4f}")
    print(f"  - Average Large Model Usage: {avg_large_model_usage:.2f}%")
    print("---------------------------------------------------------")
    print(f"Baseline: Llama-7B Only (Optimized with KV Cache):")
    print(f"  - Average Time: {avg_large_opt_time:.2f}s")
    print(f"  - Average ROUGE-L Score: {avg_large_opt_rougeL:.4f}")
    print("---------------------------------------------------------")
    print(f"Baseline: TinyLlama Only (Optimized with KV Cache):")
    print(f"  - Average Time: {avg_small_opt_time:.2f}s")
    print(f"  - Average ROUGE-L Score: {avg_small_opt_rougeL:.4f}")
    print("---------------------------------------------------------")
    print("COMPARATIVE METRICS:")
    print(f"  - Speedup vs. FAIR Baseline (Llama-7B Opt.): {fair_speedup:.2f}x")
    print(f"  - Speedup vs. NAIVE Baseline (Llama-7B no-KV): {naive_speedup:.2f}x")
    print("=========================================================")

