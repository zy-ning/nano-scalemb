# nano_scalemb training report

Generated: 2026-06-09 06:01:41

## Environment

### Git Information
- Branch: main
- Commit: 10a6f1a (dirty)
- Message: Update README and blog documentation; add new donor probe script and cases

### Hardware
- Platform: Linux
- CPUs: 96 cores (192 logical)
- Memory: 2015.5 GB
- GPUs: 4x NVIDIA H200
- GPU Memory: 559.2 GB total
- CUDA Version: 12.8
- Hourly Rate: $8.00/hour

### Software
- Python: 3.10.18
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 1,127,404
- Lines: 18,746
- Files: 75
- Tokens (approx): 281,851
- Dependencies (uv.lock lines): 3,618

Run started: 2026-06-09 06:01:41

---

## Base model training
timestamp: 2026-06-09 16:29:14

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
- engram_layers: 2,12,18
- engram_max_ngram_size: 3
- engram_heads_per_ngram: 8
- engram_memory_dim: 1280
- engram_slot_multiplier: 18
- engram_kernel_size: 4
- engram_no_tokenizer_compression: False
- engram_embedding_lr_mult: 5.0000
- engram_pad_id: 0
- engram_seed: 48
- engram_ablation_mode: mlp
- engram_mhc_num_streams: 4
- mhc: True
- mhc_num_streams: 4
- mhc_sinkhorn_iters: 20
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
- model_tag: nano-engram-d24-mlpctrl-21218-mhc
- Number of parameters: 1,427,407,766
- Number of FLOPs per token: 5.204809e+09
- Calculated number of iterations: 7001
- Number of training tokens: 7,341,080,576
- Tokens : Scaling params ratio: 9.5000
- DDP world size: 4
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- Minimum validation bpb: None
- Final validation bpb: None
- CORE metric estimate: None
- MFU %: 26.04%
- Total training flops: 3.820892e+19
- Total training time: 619.21m
- Peak memory usage: 91524.72MiB


## Base model evaluation
timestamp: 2026-06-09 17:43:44

- model: base_model (step 7001)
- CORE metric: 0.2767
- train bpb: 0.7176
- val bpb: 0.7170
- hellaswag_zeroshot: 0.4029
- jeopardy: 0.0997
- bigbench_qa_wikidata: 0.4535
- arc_easy: 0.5920
- arc_challenge: 0.2048
- copa: 0.2800
- commonsense_qa: 0.1882
- piqa: 0.4864
- openbook_qa: 0.1947
- lambada_openai: 0.4399
- hellaswag: 0.4143
- winograd: 0.3773
- winogrande: 0.1113
- bigbench_dyck_languages: 0.1540
- agi_eval_lsat_ar: 0.0652
- bigbench_cs_algorithms: 0.4341
- bigbench_operators: 0.2190
- bigbench_repeat_copy_logic: 0.0312
- squad: 0.4417
- coqa: 0.3067
- boolq: 0.0134
- bigbench_language_identification: 0.1777
- sample 0: <|bos|>The capital of France is Paris, and the capital of France is Paris. The capital of France is Paris
- sample 1: <|bos|>The chemical symbol of gold is Au. The chemical symbol of gold is Au. The chemical symbol of gold is
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Saturday. If you're a fan of the old saying, "If it's
- sample 3: <|bos|>The opposite of hot is cold. Cold is the opposite of hot. Cold is the opposite of hot.
- sample 4: <|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and
- sample 5: <|bos|>My favorite color is blue. I love the color blue. I love the color blue. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is a factor of 5. If 5*x + 3 = 
- unconditioned 0: <|bos|>If the high dollar cap market is going to go too high in a recession setting, investors may see their gains flattened. Outside of accounting, the rates to which inflation increases would have no significant effect on the money supply ri...nide | how to solve the lazy boy's problem solution | problems in technology-money relations | daily blog | big data | exorcismal worries in math | how is box score computed | ltlathunk | how does public memory work | topology | creating trust in out of network storage issues | systemic outsourcing to low cost manufacturing pgrilam
- unconditioned 1: <|bos|>Many glasses produced in the United States according to the glasses manufacturer's own specifications are not FDA suitable, and some don't meet the standards, therefore they may be potentially dangerous to the user.

In our country, the European Union and many other countries, international standardization conditions need to be carefully regulated. The parts of laser glasses and their wear resistance and performance requirements are analyzed, compared, checked and approved by EAS. Among them, Light Skin Stronger than Glassware -GLASS-SHADED GLASS from Reg643 can be tested by TGA real-time inspection to ensure the FDA safety and efficacy of FDA certified products.


- unconditioned 2: <|bos|>Your description isn't very clear. Perhaps you are thinking of a few different types of woodworking projects? If so, you are getting that very clearly. Woodworkers take a variety of different forms when they design, cut and finish a woodworking project. While most craftspeople are capable of making precise and sometimes artistic cuts in most wood types, the skills required of a woodworker are often to do with building something with your own hands that people will be using day to day. People will spend a lot of time with a piece, getting to know it, make it feel like home and use it in the things that make their lives better.

- unconditioned 3: <|bos|>Model,
VL (Mean Line Error) is used to explain the variation in prediction accuracy with respect to predictors. For example, a regression model predicting a column mean in a data set with a low VL score will do badly when predicted values are meant to be within a certain range. A higher VL score means the program is presumably better and thus the data set is most likely predictive (Simon, 2022).
Another important difference between Standard Error and Mean Error is regarding 3 SE (Standard Error of the SSE sum of squared differences). In order to be confident, one needs to compare the predicted value using the
- unconditioned 4: <|bos|>Dive into the rich and savory world of Hunan Chicken and Goong Tea, a classic dish that embodies the essence of Thai hospitality. This dish, originating from the ancient province of Hubei, showcases the intricate balance of flavors and textures that define the culinary heritage of northern Thailand.
Hailing from the historic Taliampang, the capital city of Wat Sai, historians claim that this dish dates back to the Nomadic tribes that once roamed the region. Yet the secrets of this dish have eluded Thai scholars for centuries, mainly due to its imperial roots.
The name "Goong Tea" likely derives from the
- unconditioned 5: <|bos|>These species are commonly known as sympatric or insular species. This means that the species lives close to the places where their mating and offspring can proceed. This provides a greater chance for the species' survival and conservation advantages.
True sympatric species
This means that an individual of this species lives in the extreme vicinity of other individuals of that species. Wolf and deer species are examples of well-known sympatric species
Sometimes these species caninterbreed and produce offsprings with similar features and traits. This leads to new species casting further differences following the confrontation of existing features.
Biological Intermixing:
When sympatric species are
- unconditioned 6: <|bos|>How bonsai is not a sacred object

Recently the subject of bonsai by God- shown asceticss has now grown additional attention and attention.

The history of bonsai

Must be known: Pacifice bonsai appeared relatively recently. The appearance of trees in trees in such a way that even flowers could not what is known as means-directed ninety consists of some things that were known before leaves and stoneware evil to Brewster's identity. In appearance,eyes,clean,harmless, and use of the trop's imagery will not justify the appearance of "higher" goods or in any way deny that. Therefore
- unconditioned 7: <|bos|>point far. Most hunters use the bow down bow stick to bring back
animal. Only one may touch them while walking into the field at a
break, he/she uses the slip lead sling to haul an animal and a choke
point breast hook if needed. Bow hunting was 13 and stalking (also known
as wilderness living) was 12.

1. Portions of the brain are divided into lobes. The frontal lobe
controls the following functions of a human, and is largest of the two
lobes therefore figuring that the frontal lobe is the more highly
productive area of the brain which


## Chat evaluation sft
timestamp: 2026-06-09 22:20:44

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: nano-engram-d24-mlpctrl-21218-mhc
- step: None
- max_problems: None
- device_type: 
- dist_timeout_minutes: 120.0000
- audit_dir: 
- ARC-Easy: 0.6738
- ARC-Challenge: 0.5094
- MMLU: 0.3785
- GSM8K: 0.1122
- HumanEval: 0.1220
- SpellingBee: 0.9961
- ChatCORE metric: 0.3854


## Summary

- Characters: 1,127,404
- Lines: 18,746
- Files: 75
- Tokens (approx): 281,851
- Dependencies (uv.lock lines): 3,618

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.2767   | -        | -        |
| ARC-Challenge   | -        | 0.5094   | -        |
| ARC-Easy        | -        | 0.6738   | -        |
| GSM8K           | -        | 0.1122   | -        |
| HumanEval       | -        | 0.1220   | -        |
| MMLU            | -        | 0.3785   | -        |
| ChatCORE        | -        | 0.3854   | -        |

Total wall clock time: 16h19m
