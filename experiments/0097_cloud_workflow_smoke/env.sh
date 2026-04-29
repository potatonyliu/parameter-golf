# Source this from inside the experiment folder before running.
export RUN_ID="0097_cloud_workflow_smoke"
export DATA_PATH="../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export ITERATIONS=50
export WARMUP_STEPS=0
export WARMDOWN_ITERS=1200
# Cloud workflow test: small wallclock cap forces the wallclock-branch lr_mul,
# acceptable here because we're testing the SYNC/launch flow not the schedule.
export MAX_WALLCLOCK_SECONDS=120
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
# 16384-token val cap keeps eval ~1 s. Do NOT set to 0 — full val on MPS
# takes 60-120 min per experiment (eval is called twice: pre-quant + post-
# int8-quant). Use SEED=42 re-run for marginal-result confirmation instead.
export VAL_TOKENS=16384
# Dense step logs to catch divergence early; default 200 prints only steps
# 1–10 and step 200, leaving the bulk of training invisible.
export TRAIN_LOG_EVERY=5
# Experiment-specific overrides go below:
