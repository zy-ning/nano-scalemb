# nano_scalemb training report

Generated: 2026-06-06 14:18:46

## Environment

### Git Information
- Branch: main
- Commit: ea1115c (dirty)
- Message: first commit

### Hardware
- Platform: Linux
- CPUs: 96 cores (192 logical)
- Memory: 2015.5 GB
- GPUs: 8x NVIDIA H200
- GPU Memory: 1118.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $16.00/hour

### Software
- Python: 3.10.18
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 543
- Lines: 28
- Files: 1
- Tokens (approx): 135
- Dependencies (uv.lock lines): 3,618

Run started: 2026-06-06 14:18:47

---

## Base model training
timestamp: 2026-06-06 19:25:57

- run: dummy
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 24
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- engram: True
- engram_layers: 17
- engram_max_ngram_size: 3
- engram_heads_per_ngram: 8
- engram_memory_dim: 1280
- engram_slot_multiplier: 18
- engram_kernel_size: 4
- engram_no_tokenizer_compression: False
- engram_embedding_lr_mult: 5.0000
- engram_pad_id: 0
- engram_seed: 56
- engram_ablation_mode: none
- engram_fusion_mode: mhc
- engram_mhc_num_streams: 4
- engram_mhc_sinkhorn_iters: 20
- mhc: True
- mhc_num_streams: 4
- mhc_sinkhorn_iters: 20
- mhc_head_mode: learned
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 9.5000
- device_batch_size: 8
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0040
- weight_decay: 0.2000
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- adam_beta1: 0.8000
- adam_beta2: 0.9500
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- resume_from_step: -1
- eval_every: -1
- eval_tokens: 41,943,040
- core_metric_every: -1
- core_metric_max_per_task: 500
- sample_every: -1
- save_every: -1
- model_tag: nano-scalemb-d24-1layer-17-mhc-20260605-135431
- Number of parameters: 1,951,744,804
- Number of FLOPs per token: 5.061315e+09
- Calculated number of iterations: 6784
- Number of training tokens: 7,113,539,584
- Tokens : Scaling params ratio: 9.4991
- DDP world size: 8
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- Minimum validation bpb: None
- Final validation bpb: None
- CORE metric estimate: None
- MFU %: 26.69%
- Total training flops: 3.600387e+19
- Total training time: 285.84m
- Peak memory usage: 97499.60MiB


## Base model evaluation
timestamp: 2026-06-06 20:03:08

- model: base_model (step 6784)
- CORE metric: 0.2408
- train bpb: 0.7269
- val bpb: 0.7255
- hellaswag_zeroshot: 0.3659
- jeopardy: 0.0931
- bigbench_qa_wikidata: 0.4320
- arc_easy: 0.5741
- arc_challenge: 0.1604
- copa: 0.3200
- commonsense_qa: 0.0438
- piqa: 0.4570
- openbook_qa: 0.1947
- lambada_openai: 0.4238
- hellaswag: 0.3660
- winograd: 0.2747
- winogrande: 0.0908
- bigbench_dyck_languages: 0.0870
- agi_eval_lsat_ar: 0.0652
- bigbench_cs_algorithms: 0.4159
- bigbench_operators: 0.1619
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.4123
- coqa: 0.3100
- boolq: -0.1339
- bigbench_language_identification: 0.1827
- sample 0: <|bos|>The capital of France is Paris, and the city of Paris is the capital of France. Paris is the
- sample 1: <|bos|>The chemical symbol of gold is Au, which is derived from the Latin word aurum, meaning "shining
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If yesterday was Saturday, then tomorrow will be Sunday. If yesterday was
- sample 3: <|bos|>The opposite of hot is cold. Cold is the opposite of hot. Cold is the opposite of hot.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and
- sample 5: <|bos|>My favorite color is blue. I love blue. I love blue. I love blue. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 5. If 5*x + 3 = 13, then
- unconditioned 0: <|bos|>If the high dollar amount of cash you acquire go too far in discounting life, it may appear that what you do and keep on hold your rates to embrace inflation.
If you swap your job to a nonprofit organization that provides jobs for low cost.
I notice lately it seems as if government aid starts decreasing. Is it like a reduction on my income? Can I go back on the Chapter 100 bond I was purchasing for $25,000?
As I understand it, Congress enacted a $5 billion cap on income tax with a 10% rate to protect the American pension scheme from increasing tax rates as Americans pull back
- unconditioned 1: <|bos|>Many glasses produced in the United States are made in places such as China, China, Europe, or China. These Philippine Manufacturers are, therefore, a particularly high risk of black mould growth, as typical Philippine Aleurone screen printing moulds use a spray system; Negro Wetspun parts' use an edible ink so that molding products will not be rinsed away during food preparation. Some companies claim their production line can grow mold quickly due to some equipment being exposed to weather or humidity. Other ingredients such as soap can cause it too, but it is usually thicker and can sometimes be harder to spot and mold it will
- unconditioned 2: <|bos|>Your description isn't very accurate. Even when I guess this about the color of a region it is not "the most common rich that we see". Whitish regions are less common than white regions. If you think it is, you wouldn't have three most common rich colors rather than six rich colors.
Distortion is really that big of a deal, compared to brightness variation. Bright to dark by pixel shift is going to appear different based on how far each depth buffer is from the light compared to how far it is relative to the color buffer (same pixel). Colors aren't just different brightness, they are also different amounts of yellow.

- unconditioned 3: <|bos|>Model,
On 15 March 1967 the Act was first passed in the United States demanding that the Commission set up one international Commission on teaching English-language literature (TED L), the second in the World. February 1, 1967 saw the commission break down into two "reports" issued by that Commission. One of the reports and the 30 Agreements, introduced by ENRON President, Jean Enso, was produced in a medical journal article. It stated, ""There is no textbook for English culture. It awards the student time to be intelligent, wisdomful and competitive.
The Act caused a
- unconditioned 4: <|bos|>Dive into the rich and savory land of Hunza National Park, where the convergence of extreme weather and swift mountain footsteps has come to define the desert biome.
With its Rolls Royce, India has earned the title of the most environmentally friendly nation in the world. India’s designation of Clean Air Zones (CAZ) has elevated the Status of the Air.
Orchids and buttery collards make their presence known in the annual Draba traditional festival, called weta in Kurnool district, Rajasthan, India. What’s more is that the dairy and art caring regions of Jaipur and Nairob
- unconditioned 5: <|bos|>These species are commonly known as "cape," yet to be an effect, mine involves a filling and clipping around the eyes, these eyes are "matted" to a pink behind because the head is so large compared to body. Further, when the bath salt strips an imprint of this head, it goes blurry and...doesn't show up as a pink against the stray light.
@ Catriona
No, these are not the same as "cape" except that mine involved some scalpel along the body, which will never acquire a fancy dress outfit, (maybe we could reskin Sean, after Heaven's Hollow
- unconditioned 6: <|bos|>How bonsai trees grow and live

Bonsai are one of the oldest forms of decorative art. Their history goes back to the Roman Empire, which enjoyed the practice of having dwarf trees on their homes as ornamental objects. Bonsai are also found to be native to Japan.

Here are the most common flowers explained

Some of the most common flowers are members of the primrose family called primrodiam and are known for their colourful flowers. There are also members of the sunflower family, which have a yellow coloured flower and look very similar to a sunflower yet is a member of the Primrose family of flowers.

When we say primrodiam these
- unconditioned 7: <|bos|>pointing. Most hunters use the bow downwind from their target area in a mostly
dry hole or no cover. You must hide the ball well downwind and position the bow so
that your target is a comfortable, averaged distance from you, depending on if you
are hunting downwind from the stag or upwind around them.
Know the depths in a hunting hole. Freshwater is safer, but wind
changes can make a wet hole 13 to 14 feet deep or more in the summer (and more
for wind speed) with the potential of deviating more later. Before meandering into
the


## Chat evaluation sft
timestamp: 2026-06-06 22:32:46

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: nano-scalemb-d24-1layer-17-mhc-20260605-135431
- step: None
- max_problems: None
- device_type: 
- dist_timeout_minutes: 120.0000
- audit_dir: 
- ARC-Easy: 0.6692
- ARC-Challenge: 0.5239
- MMLU: 0.3813
- GSM8K: 0.0804
- HumanEval: 0.0976
- SpellingBee: 0.9961
- ChatCORE metric: 0.3789


## Summary

- Characters: 543
- Lines: 28
- Files: 1
- Tokens (approx): 135
- Dependencies (uv.lock lines): 3,618

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.2408   | -        | -        |
| ARC-Challenge   | -        | 0.5239   | -        |
| ARC-Easy        | -        | 0.6692   | -        |
| GSM8K           | -        | 0.0804   | -        |
| HumanEval       | -        | 0.0976   | -        |
| MMLU            | -        | 0.3813   | -        |
| ChatCORE        | -        | 0.3789   | -        |

Total wall clock time: 8h13m
