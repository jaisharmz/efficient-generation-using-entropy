import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os

# --- IMPORTANT: Hugging Face Authentication ---
# To use Llama 2, you need to be authenticated.
# 1. Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and accept the terms.
# 2. In your terminal, run: `huggingface-cli login` and paste your access token.
# If you have the token as a string, you can set it here:
# HF_TOKEN = "hf_YOUR_TOKEN_HERE" 
# os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# --- 1. Model and Tokenizer Initialization ---
print("Loading models... This may take a moment, especially for the 7B model.")

# Check for CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("This script requires CUDA-enabled GPUs to run larger models.")

# We use bfloat16 for performance on A100 GPUs
dtype = torch.bfloat16

# Load the smaller, faster model ("assistant"). We upgrade from distilgpt2 to gpt2-large.
# This model is small enough to fit on one GPU.
small_model_name = "gpt2-large"
# Note: For multi-GPU, we explicitly place the small model on the first GPU.
small_model = AutoModelForCausalLM.from_pretrained(
    small_model_name, 
    torch_dtype=dtype
).to("cuda:0")
small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)


# Load the larger, more powerful model ("expert").
# device_map="auto" will spread this model across all available GPUs (your 2xA100s).
large_model_name = "meta-llama/Llama-2-7b-chat-hf"
large_model = AutoModelForCausalLM.from_pretrained(
    large_model_name,
    torch_dtype=dtype,
    device_map="auto"  # This is the key for multi-GPU inference
)
large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)

# --- Tokenizer Unification ---
# Since we are using two different models, they have different tokenizers.
# To avoid errors, we'll use the more capable (larger) tokenizer for all operations.
# We must ensure the pad_token is set for the chosen tokenizer.
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


# --- 2. Entropy Calculation Function (No changes needed) ---
def calculate_entropy(logits):
    probabilities = F.softmax(logits, dim=-1)
    epsilon = 1e-9
    entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=-1)
    return entropy.item()


# --- 3. Gated Generation Logic ---
def generate_with_entropy_gate(
    prompt, max_new_tokens=50, entropy_threshold=3.5
):
    print(f"\n--- Starting Generation ---")
    print(f"Prompt: '{prompt}'")
    print(f"Entropy Threshold: {entropy_threshold}\n")
    
    # Use the unified (large) tokenizer to encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    generated_sequence = input_ids.clone()

    for i in range(max_new_tokens):
        with torch.no_grad():
            # --- Step A: Run the small model first ---
            # Ensure the input is on the same device as the small model
            small_model_outputs = small_model(input_ids=generated_sequence.to(small_model.device))
            next_token_logits_small = small_model_outputs.logits[:, -1, :]

            # --- Step B: Calculate entropy ---
            entropy = calculate_entropy(next_token_logits_small)

            chosen_model = ""
            # --- Step C: The Gating Decision ---
            if entropy < entropy_threshold:
                chosen_model = "Small (Low Entropy)"
                next_token_logits = next_token_logits_small
            else:
                chosen_model = "Large (HIGH ENTROPY)"
                # For the large model, inputs are automatically handled by accelerate
                large_model_outputs = large_model(input_ids=generated_sequence)
                next_token_logits = large_model_outputs.logits[:, -1, :]

            # --- Step D: Sample the next token ---
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Append to sequence (must be on the same device for cat)
            generated_sequence = torch.cat([generated_sequence, next_token_id.to(generated_sequence.device)], dim=-1)

            # Decode the newly generated token for logging
            new_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(
                f"Step {i+1:2d} | Entropy: {entropy:6.3f} | Model: {chosen_model:<20} | Token: '{new_token_text}'"
            )

    # Decode the final full sequence using the unified tokenizer
    final_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    return final_text


# --- 4. Running the Example ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # The old entropy thresholds will NOT work well. The output distribution
    # of gpt2-large is very different from distilgpt2. You will need to
    # experiment to find a new, useful threshold. Let's start with a low guess.
    
    print("\n\n=========================================================")
    print("========== Example 1: Factual Knowledge Prompt =========")
    print("=========================================================")
    # Llama-2 chat models work best with a specific format.
    factual_prompt = "[INST] The capital of France is Paris, and the capital of Japan is [/INST]"

    # With a higher threshold, the small model may fail.
    final_output_factual_high = generate_with_entropy_gate(
        factual_prompt, max_new_tokens=5, entropy_threshold=3.0
    )
    print("\n--- Final Output (High Threshold - more small model) ---")
    print(final_output_factual_high)
    
    # With a lower threshold, we force a switch to the expert model.
    final_output_factual_low = generate_with_entropy_gate(
        factual_prompt, max_new_tokens=5, entropy_threshold=1.5
    )
    print("\n--- Final Output (Low Threshold - more large model) ---")
    print(final_output_factual_low)
    
    print("\n\n=========================================================")
    print("============ Example 2: Code Generation Prompt ===========")
    print("=========================================================")
    code_prompt = "[INST] Write a python function to compute the Nth fibonacci number. [/INST]\n```python\ndef fibonacci(n):"
    
    # Let's see how the small model struggles with code.
    final_output_code_high = generate_with_entropy_gate(
        code_prompt, max_new_tokens=50, entropy_threshold=4.0
    )
    print("\n--- Final Output (High Threshold - more small model) ---")
    print(final_output_code_high)

    # A stricter threshold should yield better code.
    final_output_code_low = generate_with_entropy_gate(
        code_prompt, max_new_tokens=50, entropy_threshold=2.0
    )
    print("\n--- Final Output (Low Threshold - more large model) ---")
    print(final_output_code_low)

