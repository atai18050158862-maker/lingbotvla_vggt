#!/bin/bash
# Prepare RoboTwin 2.0 aloha-agilex_clean_50 data (50 tasks) for LingBot-VLA training.
#
# Input  : ${DATA_ROOT}/<task>/.../aloha-agilex_clean_50/...  The raw episode dir (the one holding
#          data/episode*.hdf5 and instructions/) is searched up to 4 levels below the task dir,
#          so an extra folder level left by unzipping is fine.
# Output : ${LEROBOT_ROOT}/<task>-aloha-agilex_clean_50-50   (LeRobot v3.0, one dataset per task;
#                                                              RoboTwin writes v2.1, upgraded in step 3,
#                                                              original kept as <name>_old)
#          assets/training_data/robotwin_clean50.txt          (training list, written only when all
#                                                              50 tasks are ready)
#
# Per task:
#   - already converted into LEROBOT_ROOT   -> skipped, so the script can be re-run after a failure
#   - already LeRobot (has meta/info.json)  -> used as is
#   - raw RoboTwin (has data/*.hdf5)        -> converted with RoboTwin's official pi0 pipeline
#                                              (scripts/process_data.py, then generate.sh)
#
# Usage (from the repo root; every path below can be overridden with the same-named env var):
#   CHECK_ONLY=1 bash scripts/prepare_robotwin_clean50.sh   # report what was found, change nothing
#   bash scripts/prepare_robotwin_clean50.sh                # convert + write list
#
# DATA_ROOT is only read. Converted data goes to LEROBOT_ROOT, and
# ROBOTWIN_ROOT/data/<task>/aloha-agilex_clean_50 becomes a symlink to the raw episode dir.
#
# Requirements for raw conversion:
#   - RoboTwin repo at commit 13c3c47ff4312dd62484bcd51be034af55c062d1, with
#     `cd policy/pi0 && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync` already done
#   - ${PYTHON} (default: python) has h5py, opencv-python and numpy (the lingbotvla env does)
set -euo pipefail

# Prefer the active conda env's python over system ones earlier on PATH.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

USER_ROOT=/efs/share/1919650160032350208/users/huangzt71
DATA_ROOT=${DATA_ROOT:-/efs/share/1919650160032350208/projects/foundation_model/foundation_model_/datasets/official_robotwin2/dataset_unziped}
LEROBOT_ROOT=${LEROBOT_ROOT:-${USER_ROOT}/datasets/robotwin2_clean50_lerobot}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-${USER_ROOT}/RoboTwin}
PYTHON=${PYTHON:-python}
CHECK_ONLY=${CHECK_ONLY:-0}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
SETTING=aloha-agilex_clean_50
EXPERT_NUM=50
ROBOT_CONFIG_NAME=robotwin

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIST_FILE=${LIST_FILE:-${REPO_ROOT}/assets/training_data/robotwin_clean50.txt}

TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock
  click_bell dump_bin_bigbin grab_roller handover_block handover_mic
  hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away
  move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
  place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries
  place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
  place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand
  place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet
  rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three
  stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the raw episode dir (parent of data/*.hdf5) for SETTING under task dir $1.
find_raw_src() {
  local d
  while IFS= read -r d; do
    if compgen -G "${d}/*.hdf5" > /dev/null; then
      dirname "${d}"
      return 0
    fi
  done < <(find -L "$1" -maxdepth 4 -type d -name data -path "*${SETTING}*" 2> /dev/null | sort)
  return 1
}

# Prints a LeRobot dataset root (parent of meta/info.json) for SETTING under task dir $1.
find_lerobot_src() {
  local f
  f=$(find -L "$1" -maxdepth 4 -type f -path "*${SETTING}*/meta/info.json" 2> /dev/null | sort | head -n 1)
  [ -n "${f}" ] || return 1
  dirname "$(dirname "${f}")"
}

[ -d "${DATA_ROOT}" ] || die "DATA_ROOT not found: ${DATA_ROOT}"

# ---------- 1. classify every task (FORMAT[i] / SRC[i] belong to TASKS[i]) ----------
FORMAT=()
SRC=()
n_lerobot=0; n_raw=0; n_done=0; n_missing=0
first_missing_dir=""
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  task_dir="${DATA_ROOT}/${task}"
  out="${LEROBOT_ROOT}/${task}-${SETTING}-${EXPERT_NUM}"
  SRC[$i]=""
  if [ -f "${out}/meta/info.json" ]; then
    FORMAT[$i]=done; n_done=$((n_done + 1))
  elif src=$(find_lerobot_src "${task_dir}"); then
    FORMAT[$i]=lerobot; SRC[$i]="${src}"; n_lerobot=$((n_lerobot + 1))
  elif src=$(find_raw_src "${task_dir}"); then
    # process_data.py opens data/episode{0..N-1}.hdf5 and instructions/episode{0..N-1}.json
    absent=""
    for ((e = 0; e < EXPERT_NUM; e++)); do
      [ -f "${src}/data/episode${e}.hdf5" ] || absent="${absent} data/episode${e}.hdf5"
      [ -f "${src}/instructions/episode${e}.json" ] || absent="${absent} instructions/episode${e}.json"
    done
    if [ -z "${absent}" ]; then
      FORMAT[$i]=raw; SRC[$i]="${src}"; n_raw=$((n_raw + 1))
    else
      FORMAT[$i]=missing; n_missing=$((n_missing + 1))
      echo "MISSING [${task}]: incomplete episodes in ${src}:$(echo "${absent}" | cut -c1-200)"
    fi
  else
    FORMAT[$i]=missing; n_missing=$((n_missing + 1))
    if [ ! -d "${task_dir}" ]; then
      echo "MISSING [${task}]: task dir not found: ${task_dir}"
    elif zip=$(find -L "${task_dir}" -maxdepth 3 -name "*${SETTING}*.zip" 2> /dev/null | head -n 1) && [ -n "${zip}" ]; then
      echo "MISSING [${task}]: still zipped, unzip first: ${zip}"
    else
      echo "MISSING [${task}]: no */${SETTING}*/data/*.hdf5 within 4 levels of ${task_dir}"
      [ -n "${first_missing_dir}" ] || first_missing_dir="${task_dir}"
    fi
  fi
done

if [ -n "${first_missing_dir}" ]; then
  echo "---- layout of ${first_missing_dir} (3 levels) ----"
  find -L "${first_missing_dir}" -maxdepth 3 2> /dev/null | head -n 30
fi
if [ "${n_missing}" -eq "${#TASKS[@]}" ]; then
  echo "---- first entries of ${DATA_ROOT} ----"
  ls -1 "${DATA_ROOT}" | head -n 20
fi
for i in "${!TASKS[@]}"; do
  if [ -n "${SRC[$i]}" ]; then
    echo "example: ${TASKS[$i]} -> ${SRC[$i]} (${FORMAT[$i]})"
    break
  fi
done
echo "tasks: ${#TASKS[@]} | converted earlier: ${n_done} | already LeRobot: ${n_lerobot} | raw to convert: ${n_raw} | missing: ${n_missing}"

if [ "${CHECK_ONLY}" = "1" ]; then
  exit 0
fi
if [ "${n_missing}" -gt 0 ] && [ "${ALLOW_PARTIAL}" != "1" ]; then
  die "${n_missing} task(s) missing; nothing converted and ${LIST_FILE} left unchanged. Fix them (or set ALLOW_PARTIAL=1) and re-run."
fi

# ---------- 2. convert raw tasks ----------
if [ "${n_raw}" -gt 0 ]; then
  [ -d "${ROBOTWIN_ROOT}" ] || die "RoboTwin repo not found at ${ROBOTWIN_ROOT}; clone it there or set ROBOTWIN_ROOT=..."
  PI0_DIR="${ROBOTWIN_ROOT}/policy/pi0"
  [ -f "${PI0_DIR}/scripts/process_data.py" ] || die "Not found: ${PI0_DIR}/scripts/process_data.py"
  [ -f "${PI0_DIR}/generate.sh" ] || die "Not found: ${PI0_DIR}/generate.sh"
  [ -d "${PI0_DIR}/.venv" ] || die "RoboTwin pi0 env missing: run 'cd ${PI0_DIR} && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync'"
  command -v uv > /dev/null || die "uv not on PATH: pip install uv"
  if ! import_err=$("${PYTHON}" -c "import h5py, cv2, numpy" 2>&1); then
    die "'${PYTHON}' ($(command -v "${PYTHON}" || echo 'not found')) cannot import h5py/cv2/numpy: $(echo "${import_err}" | tail -n 1)
       Activate the training env (conda activate lingbotvla) or set PYTHON=/path/to/python."
  fi

  mkdir -p "${LEROBOT_ROOT}" "${ROBOTWIN_ROOT}/data" "${PI0_DIR}/processed_data"
  default_lerobot_home="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/lerobot"

  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    [ "${FORMAT[$i]}" = "raw" ] || continue
    repo_id="${task}-${SETTING}-${EXPERT_NUM}"
    out="${LEROBOT_ROOT}/${repo_id}"
    echo "==== [${task}] converting ${SRC[$i]} -> ${out}"

    # process_data.py reads ../../data/<task>/<setting>, i.e. ${ROBOTWIN_ROOT}/data/<task>/<setting>
    task_link_dir="${ROBOTWIN_ROOT}/data/${task}"
    [ ! -L "${task_link_dir}" ] || die "${task_link_dir} is a symlink; remove it (rm ${task_link_dir}) and re-run."
    mkdir -p "${task_link_dir}"
    link="${task_link_dir}/${SETTING}"
    if [ -e "${link}" ] || [ -L "${link}" ]; then
      [ "$(readlink -f "${link}")" = "$(readlink -f "${SRC[$i]}")" ] \
        || die "${link} exists and does not point to ${SRC[$i]}; move it away first."
    else
      ln -s "${SRC[$i]}" "${link}"
    fi

    (
      cd "${PI0_DIR}"
      rm -rf "processed_data/${repo_id}"  # stale intermediate from an interrupted run
      "${PYTHON}" scripts/process_data.py "${task}" "${SETTING}" "${EXPERT_NUM}"
      HF_LEROBOT_HOME="${LEROBOT_ROOT}" bash generate.sh "processed_data/${repo_id}" "${repo_id}"
    )

    # Older lerobot ignores HF_LEROBOT_HOME and writes to the default cache instead.
    if [ ! -f "${out}/meta/info.json" ] && [ -f "${default_lerobot_home}/${repo_id}/meta/info.json" ]; then
      mv "${default_lerobot_home}/${repo_id}" "${out}"
    fi
    [ -f "${out}/meta/info.json" ] || die "[${task}] conversion finished but ${out}/meta/info.json not found."
    FORMAT[$i]=done
  done
fi

# ---------- 3. upgrade LeRobot v2.1 -> v3.0 ----------
# RoboTwin's pi0 converter writes v2.1, while the training env ships lerobot 0.4.2, which
# only reads v3.0. lerobot's own converter upgrades in place: the result replaces
# LEROBOT_ROOT/<repo_id> and the original is kept as LEROBOT_ROOT/<repo_id>_old.
codebase_version() {
  "${PYTHON}" -c "import json, sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$1/meta/info.json"
}
lerobot_ready=0
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  case "${FORMAT[$i]}" in
    done) path="${LEROBOT_ROOT}/${task}-${SETTING}-${EXPERT_NUM}" ;;
    lerobot) path="${SRC[$i]}" ;;
    *) continue ;;
  esac
  version=$(codebase_version "${path}")
  [ "${version}" = "v3.0" ] && continue
  [ "${FORMAT[$i]}" = "done" ] || die "[${task}] ${path} is LeRobot ${version} inside read-only DATA_ROOT; copy it under LEROBOT_ROOT first."
  if [ "${lerobot_ready}" = "0" ]; then
    "${PYTHON}" -c "import lerobot.datasets.v30.convert_dataset_v21_to_v30" 2> /dev/null \
      || die "'${PYTHON}' has no lerobot v3 converter; activate the training env (it ships lerobot 0.4.2)."
    lerobot_ready=1
  fi
  echo "==== [${task}] upgrading LeRobot ${version} -> v3.0: ${path}"
  HF_HUB_OFFLINE=1 "${PYTHON}" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id="$(basename "${path}")" --root="$(dirname "${path}")" --push-to-hub=false
  version=$(codebase_version "${path}")
  [ "${version}" = "v3.0" ] || die "[${task}] upgrade finished but ${path} reports codebase_version ${version}."
done

# ---------- 4. write the training list ----------
tmp_list=$(mktemp)
n_listed=0
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  case "${FORMAT[$i]}" in
    lerobot) path="${SRC[$i]}" ;;
    done) path="${LEROBOT_ROOT}/${task}-${SETTING}-${EXPERT_NUM}" ;;
    *) continue ;;
  esac
  echo "${ROBOT_CONFIG_NAME} ${path}" >> "${tmp_list}"
  n_listed=$((n_listed + 1))
done
if [ "${n_listed}" -eq 0 ]; then
  rm -f "${tmp_list}"
  die "No dataset ready; ${LIST_FILE} left unchanged."
fi
mv "${tmp_list}" "${LIST_FILE}"
echo "Wrote ${n_listed}/${#TASKS[@]} datasets to ${LIST_FILE}"
if compgen -G "${LEROBOT_ROOT}/*_old" > /dev/null; then
  echo "Pre-upgrade v2.1 copies remain as ${LEROBOT_ROOT}/*_old; delete them once training reads the data fine."
fi
[ "${n_listed}" -eq "${#TASKS[@]}" ] || echo "WARNING: ${#TASKS[@]} tasks expected (ALLOW_PARTIAL=1 was set)."
