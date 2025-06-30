import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# --- 1. Model and Tokenizer Initialization ---
# We'll use distilgpt2 as our fast, small model and gpt2 as our slow, large model.
# In a real-world scenario, the performance gap would be much larger
# (e.g., a 1B vs a 70B model).
# For this example, we use models that are easy to run locally.

print("Loading models... This may take a moment.")

# It's recommended to run this on a CUDA-enabled machine for faster inference.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the smaller, faster model ("assistant")
small_model_name = "distilgpt2"
small_model = AutoModelForCausalLM.from_pretrained(small_model_name).to(device)
small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)

# Load the larger, more powerful model ("expert")
large_model_name = "gpt2"
large_model = AutoModelForCausalLM.from_pretrained(large_model_name).to(device)
large_tokenizer = AutoTokenizer.from_pretrained(large_model_name)

# Ensure tokenizers have a pad token (for batching, if needed, and consistency)
if small_tokenizer.pad_token is None:
    small_tokenizer.pad_token = small_tokenizer.eos_token
if large_tokenizer.pad_token is None:
    large_tokenizer.pad_token = large_tokenizer.eos_token

print("Models loaded successfully.")


# --- 2. Entropy Calculation Function ---
def calculate_entropy(logits):
    """
    Calculates the entropy of a probability distribution given the logits.
    Entropy is a measure of uncertainty. High entropy means the model
    is uncertain about the next token.
    Formula: H(X) = - sum(p(x) * log2(p(x)))
    """
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Use a small epsilon to avoid log(0) which is undefined.
    epsilon = 1e-9

    # Calculate entropy
    # We use log2 for the conventional "bits" unit of entropy.
    entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=-1)

    return entropy.item()


# --- 3. Gated Generation Logic ---
def generate_with_entropy_gate(
    prompt, max_new_tokens=50, entropy_threshold=3.5
):
    """
    Generates text by using the small model's entropy to decide when to switch
    to the large model.

    Args:
        prompt (str): The initial text to start generation from.
        max_new_tokens (int): The number of new tokens to generate.
        entropy_threshold (float): The entropy value above which we switch to the large model.
                                   This is a hyperparameter you would tune. A lower
                                   threshold means switching more often.
    """
    print(f"\n--- Starting Generation ---")
    print(f"Prompt: '{prompt}'")
    print(f"Entropy Threshold: {entropy_threshold}\n")

    # Use the same tokenizer for encoding the initial prompt.
    # We assume vocabularies are compatible, which they are for gpt2/distilgpt2.
    input_ids = small_tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_sequence = input_ids.clone()

    for i in range(max_new_tokens):
        # We don't need gradients, so we use torch.no_grad() for efficiency
        with torch.no_grad():
            # --- Step A: Run the small model first ---
            small_model_outputs = small_model(input_ids=generated_sequence)
            next_token_logits_small = small_model_outputs.logits[:, -1, :]

            # --- Step B: Calculate the entropy of the small model's prediction ---
            entropy = calculate_entropy(next_token_logits_small)

            chosen_model = ""
            # --- Step C: The Gating Decision ---
            if entropy < entropy_threshold:
                # If entropy is low, we trust the small model.
                chosen_model = "Small (Low Entropy)"
                next_token_logits = next_token_logits_small
            else:
                # If entropy is high, the small model is uncertain.
                # We discard its result and use the large model for this token.
                chosen_model = "Large (HIGH ENTROPY)"
                large_model_outputs = large_model(input_ids=generated_sequence)
                next_token_logits = large_model_outputs.logits[:, -1, :]

            # --- Step D: Sample the next token from the chosen distribution ---
            # We use simple greedy sampling (argmax) for deterministic results here.
            # In practice, you might use top-k or nucleus sampling.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Append the new token to our sequence
            generated_sequence = torch.cat([generated_sequence, next_token_id], dim=-1)

            # Decode the newly generated token for logging
            new_token_text = small_tokenizer.decode(next_token_id[0])
            print(
                f"Step {i+1:2d} | Entropy: {entropy:6.3f} | Model: {chosen_model:<20} | Token: '{new_token_text}'"
            )

    # Decode the final full sequence
    final_text = small_tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    return final_text


# --- 4. Running the Example ---
if __name__ == "__main__":
    # A prompt where the model might be certain at first, then less certain.
    test_prompt = "The old lighthouse stood on the cliff, its light a beacon against the"

    # --- Run with a threshold that allows the small model to work more ---
    # A higher threshold means we are more tolerant of the small model's uncertainty.
    final_output_high_thresh = generate_with_entropy_gate(
        test_prompt, max_new_tokens=20, entropy_threshold=4.0
    )
    print("\n--- Final Output (High Threshold) ---")
    print(final_output_high_thresh)

    # --- Run with a threshold that invokes the large model more often ---
    # A lower threshold means we are less tolerant and switch to the expert sooner.
    final_output_low_thresh = generate_with_entropy_gate(
        test_prompt, max_new_tokens=20, entropy_threshold=3.0
    )
    print("\n--- Final Output (Low Threshold) ---")
    print(final_output_low_thresh)

