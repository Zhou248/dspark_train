# Qwen3.6-35B-A3B multimodal DSpark training

## Scripts

1. `common_env.sh`: shared model, dataset, NPU, and stability settings.
2. `01_build_multimodal_dataset.py`: convert parquet/JSON/JSONL data into the
   multimodal `conversations.jsonl` consumed by msModelSpec.
3. `02_prepare_data.sh`: normalize raw data, run `prepare_data.py`, and create a
   1-D Qwen3 draft decoder config.
4. `02a_build_draft_config.py`: build that decoder config without copying the
   verifier's multimodal MRoPE fields.
5. `03_launch_vllm_target.sh`: launch Qwen3.6 on vLLM-Ascend in
   `extract_hidden_states` mode. Keep this process running during 04/04a.
6. `04_train_dspark.sh`: online training; hidden states are generated on demand
   and deleted after each sample.
7. `04a_generate_hidden_states.sh`: offline-data stage; generate and validate
   all hidden-state files before training.
8. `repair_hidden_states.py`: validate existing `hs_*.safetensors` and remove
   files containing NaN/Inf or mismatched token IDs.
9. `04b_train_offline.sh`: train only from validated cached hidden states.
10. `05_convert_and_serve.sh`: serve `checkpoint_best` directly with DSpark.

## Recommended offline workflow

```bash
bash 02_prepare_data.sh
bash 03_launch_vllm_target.sh       # terminal A; leave it running
bash 04a_generate_hidden_states.sh  # terminal B
# Stop terminal A after every sample has a valid hidden-state file.
bash 04b_train_offline.sh
bash 05_convert_and_serve.sh
```

Hidden-state extraction defaults to the conservative configuration
`QD_MAX_NUM_SEQS=1`, `HS_CONCURRENCY=1`, synchronous scheduling, and AIV off.
After a clean run, increase one dimension at a time:

```bash
QD_MAX_NUM_SEQS=2 HS_CONCURRENCY=2 bash 03_launch_vllm_target.sh
HS_CONCURRENCY=2 bash 04a_generate_hidden_states.sh
```

Do not increase both parameters until the previous setting has completed a
representative multimodal sample set without NaN or Inf values.
