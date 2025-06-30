# efficient-generation-using-entropy

Quickstart:
```
python main.py
```

Some example results:
```
Loading models... This may take a moment, especially for the 7B model.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.98s/it]
Models loaded successfully.
Small model (TinyLlama/TinyLlama-1.1B-Chat-v1.0) is on: cuda:0
Large model (meta-llama/Llama-2-7b-chat-hf) device map: {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1}

--- Calibrating entropy threshold from 'wikitext' ---
Target percentile: 50% (will trigger large model on the top 50% of tokens)
Processing dataset samples to gather entropy distribution...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:06<00:00, 14.46it/s]
Calibration complete. Calculated Threshold at 50th percentile: 2.1406


=========================================================
====== Evaluating Gated Generation on Test Prompts ======
=========================================================

--- Processing Prompt 1/5 ---
Gated Time: 3.15s | Large Model Time: 5.20s | Small Model Time: 0.06s
Gated ROUGE-L: 0.3077 | Large ROUGE-L: 0.3750 | Small ROUGE-L: 0.0000

--- Processing Prompt 2/5 ---
Gated Time: 2.14s | Large Model Time: 5.02s | Small Model Time: 0.05s
Gated ROUGE-L: 0.3373 | Large ROUGE-L: 0.3333 | Small ROUGE-L: 0.0000

--- Processing Prompt 3/5 ---
Gated Time: 6.18s | Large Model Time: 5.13s | Small Model Time: 0.06s
Gated ROUGE-L: 0.2222 | Large ROUGE-L: 0.2609 | Small ROUGE-L: 0.0000

--- Processing Prompt 4/5 ---
Gated Time: 1.71s | Large Model Time: 4.99s | Small Model Time: 0.04s
Gated ROUGE-L: 0.1867 | Large ROUGE-L: 0.2169 | Small ROUGE-L: 0.0000

--- Processing Prompt 5/5 ---
Gated Time: 4.05s | Large Model Time: 5.01s | Small Model Time: 0.04s
Gated ROUGE-L: 0.2651 | Large ROUGE-L: 0.2927 | Small ROUGE-L: 0.0000


=========================================================
============== AGGREGATE PERFORMANCE METRICS ==============
=========================================================
Gated Method (TinyLlama + Llama-7B):
  - Average Time: 3.45s
  - Average ROUGE-L Score: 0.2638
  - Average Large Model Usage: 19.20%
---------------------------------------------------------
Baseline: Llama-7B Only (Optimized with KV Cache):
  - Average Time: 5.07s
  - Average ROUGE-L Score: 0.2958
---------------------------------------------------------
Baseline: TinyLlama Only (Optimized with KV Cache):
  - Average Time: 0.05s
  - Average ROUGE-L Score: 0.0000
---------------------------------------------------------
COMPARATIVE METRICS:
  - Speedup vs. FAIR Baseline (Llama-7B Opt.): 1.47x
  - Speedup vs. NAIVE Baseline (Llama-7B no-KV): 2.35x
=========================================================
```