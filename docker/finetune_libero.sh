#!/bin/bash
# Fine-tune pi0 on LIBERO datasets
# Run inside the container:
#   docker run --rm --runtime=nvidia ... -v /opt/checkpoints:/opt/openpi/checkpoints sandbox:latest /finetune_libero.sh
set -euo pipefail

OPENPI_DIR="/opt/openpi"
RLDS_DATA_DIR="${RLDS_DATA_DIR:-/data/libero_rlds}"
EXP_NAME="${EXP_NAME:-libero_ft}"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_low_mem_finetune}"
# HF_LEROBOT_HOME: where LeRobot datasets are stored locally
# The training config expects repo_id="physical-intelligence/libero"
# so the dataset must be at $HF_LEROBOT_HOME/physical-intelligence/libero
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/lerobot}"

echo "================================================================"
echo " pi0 Fine-tuning on LIBERO"
echo " Config         : ${CONFIG_NAME}"
echo " Exp name       : ${EXP_NAME}"
echo " RLDS data dir  : ${RLDS_DATA_DIR}"
echo " LeRobot home   : ${HF_LEROBOT_HOME}"
echo "================================================================"

# ── 0. Install tensorflow + tensorflow_datasets ───────────────────────────────
echo ""
echo "=== Step 0: Install tensorflow + tensorflow_datasets ==="
conda run --no-capture-output -n pi0 pip install -q tensorflow tensorflow_datasets

# ── 1. Download RLDS datasets from HuggingFace ───────────────────────────────
echo ""
echo "=== Step 1: Download LIBERO RLDS datasets from HuggingFace ==="
# Source: https://huggingface.co/datasets/openvla/modified_libero_rlds
# Contains: libero_10_no_noops / libero_goal_no_noops / libero_object_no_noops / libero_spatial_no_noops
mkdir -p "${RLDS_DATA_DIR}"

if [ -d "${RLDS_DATA_DIR}/libero_10_no_noops" ] && [ "$(ls -A ${RLDS_DATA_DIR}/libero_10_no_noops 2>/dev/null)" ]; then
    echo "[skip] RLDS datasets already downloaded"
else
    echo "[download] openvla/modified_libero_rlds from HuggingFace (~40GB)..."
    conda run --no-capture-output -n pi0 huggingface-cli download openvla/modified_libero_rlds \
        --repo-type dataset \
        --local-dir "${RLDS_DATA_DIR}"
fi

# ── 2. Convert to LeRobot format ─────────────────────────────────────────────
echo ""
echo "=== Step 2: Convert LIBERO RLDS → LeRobot format ==="

LEROBOT_DATASET="${HF_LEROBOT_HOME}/physical-intelligence/libero"
if [ -d "${LEROBOT_DATASET}" ] && [ "$(ls -A ${LEROBOT_DATASET} 2>/dev/null)" ]; then
    echo "[skip] LeRobot dataset already exists at ${LEROBOT_DATASET}"
else
    echo "[convert] Running data conversion (~30 min)..."
    # Write a wrapper that patches REPO_NAME to match the pi05_libero training config
    cat > /tmp/run_convert.py << 'PYEOF'
import sys, os
sys.path.insert(0, '/opt/openpi')
os.chdir('/opt/openpi')

# Monkey-patch REPO_NAME before the module sets it
import examples.libero.convert_libero_data_to_lerobot as m
m.REPO_NAME = 'physical-intelligence/libero'

import tyro
tyro.cli(m.main)
PYEOF
    conda run --no-capture-output -n pi0 bash -c "
        export HF_LEROBOT_HOME='${HF_LEROBOT_HOME}'
        python /tmp/run_convert.py --data_dir '${RLDS_DATA_DIR}'
    "
fi

# ── 2.5. Uninstall TensorFlow (no longer needed; conflicts with protobuf) ─────
echo ""
echo "=== Step 2.5: Remove TensorFlow to avoid protobuf conflict ==="
conda run --no-capture-output -n pi0 pip uninstall -y tensorflow tensorflow_datasets 2>/dev/null || true

# ── 2.7. Inject config + patch train.py for richer wandb logging ──────────────
echo ""
echo "=== Step 2.7: Inject config + patch train.py (wandb: LR, per-layer grads, periodic images) ==="

# Patch train.py to add richer wandb logging
conda run --no-capture-output -n pi0 python3 << 'PYEOF'
train_path = '/opt/openpi/scripts/train.py'
with open(train_path) as f:
    text = f.read()

if '# WANDB_RICH_LOGGING' not in text:
    # Add LR logging + periodic camera images every 2000 steps
    old_log = '''            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)'''
    new_log = '''            # WANDB_RICH_LOGGING: add learning rate
            try:
                schedule_fn = config.lr_schedule.create()
                reduced_info["learning_rate"] = float(schedule_fn(step))
            except Exception:
                pass
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)
        # WANDB_RICH_LOGGING: log sample camera images every 2000 steps
        if step > 0 and step % 2000 == 0:
            try:
                cam_strip = np.concatenate(
                    [np.array(img[0]) for img in batch[0].images.values()], axis=1
                )
                wandb.log({"train/sample_cameras": wandb.Image(cam_strip, caption=f"step {step}")}, step=step)
            except Exception:
                pass'''
    if old_log in text:
        text = text.replace(old_log, new_log)
        with open(train_path, 'w') as f:
            f.write(text)
        print('train.py patched: LR + camera images every 2000 steps')
    else:
        print('WARNING: Could not find insertion point in train.py')
else:
    print('train.py already patched')
PYEOF

conda run --no-capture-output -n pi0 python3 << 'PYEOF'
import sys
sys.path.insert(0, '/opt/openpi/src')
config_path = '/opt/openpi/src/openpi/training/config.py'
with open(config_path) as f:
    text = f.read()
if 'pi05_libero_low_mem_finetune' not in text:
    new_cfg = '''ベス
    TrainConfig(
        name="pi05_libero_low_mem_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                                    paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),'''
    marker = '#\n    # Fine-tuning Aloha configs.'
    text = text.replace(marker, new_cfg + '\n    ' + marker.lstrip())
    with open(config_path, 'w') as f:
        f.write(text)
    print('Config pi05_libero_low_mem_finetune injected')
else:
    print('Config pi05_libero_low_mem_finetune already present')
PYEOF

# ── 3. Compute normalization stats ────────────────────────────────────────────
echo ""
echo "=== Step 3: Compute normalization stats ==="

NORM_STATS_PATH="${OPENPI_DIR}/assets/${CONFIG_NAME}/physical-intelligence/libero/norm_stats.json"
# Also accept pre-computed pi05_libero norm stats (same dataset, same model type)
FALLBACK_STATS="${OPENPI_DIR}/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
if [ -f "${NORM_STATS_PATH}" ]; then
    echo "[skip] norm stats already exist at ${NORM_STATS_PATH}"
elif [ -f "${FALLBACK_STATS}" ] && [[ "${CONFIG_NAME}" == *"libero_low_mem"* ]]; then
    echo "[reuse] copying pi05_libero norm stats for ${CONFIG_NAME}..."
    mkdir -p "$(dirname ${NORM_STATS_PATH})"
    cp "${FALLBACK_STATS}" "${NORM_STATS_PATH}"
else
    echo "[compute] norm stats for ${CONFIG_NAME}..."
    conda run --no-capture-output -n pi0 bash -c "
        export HF_LEROBOT_HOME='${HF_LEROBOT_HOME}'
        cd ${OPENPI_DIR}
        GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=/opt/conda/envs/pi0 uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
    "
fi

# ── 4. Train ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Training ==="
echo "  Checkpoints → ${OPENPI_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/"
echo "  Steps       : 30,000 (default)"
echo "  (Ctrl+C to stop; checkpoints saved periodically)"
echo ""

conda run --no-capture-output -n pi0 bash -c "
    export HF_LEROBOT_HOME='${HF_LEROBOT_HOME}'
    cd ${OPENPI_DIR}
    XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9} \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/conda/envs/pi0 \
    PYTHONUNBUFFERED=1 \
    uv run scripts/train.py ${CONFIG_NAME} \
        --exp-name=${EXP_NAME} \
        --overwrite \
        --batch-size=${BATCH_SIZE:-2}
"
