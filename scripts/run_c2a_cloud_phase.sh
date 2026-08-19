#!/usr/bin/env bash
set -uo pipefail

phase="${1:-}"
case "$phase" in
  prepare|scan|review|research) ;;
  *)
    echo "phase 必须是 prepare、scan、review 或 research" >&2
    exit 2
    ;;
esac

project_root="${ROBOT_QUANT_PROJECT_ROOT:-/srv/robot-quant}"
python_bin="${ROBOT_QUANT_PYTHON:-${project_root}/.venv/bin/python}"
lock_file="${ROBOT_QUANT_LOCK_FILE:-${project_root}/.c2a-cloud.lock}"
trade_day="$(TZ=Asia/Shanghai date +%F)"

cd "$project_root" || {
  echo "C2-A 项目目录不可访问" >&2
  exit 2
}
project_root="$(pwd -P)"
failure_recorder="${project_root}/scripts/record_c2a_failure.py"
trusted_head=""

phase_paths() {
  paths=(
    "data/c2a_results/cloud_${phase}_latest.json"
    "data/c2a_results/${phase}/${trade_day}.json"
    "reports/c2a_${phase}_latest.md"
    "reports/c2a/${trade_day}-${phase}.md"
  )

  if [[ "$phase" == "scan" ]]; then
    paths+=(
      "data/c2a_results/fast_latest.json"
      "data/c2a_results/intraday/${trade_day}.json"
    )
  elif [[ "$phase" == "research" ]]; then
    paths+=(
      "data/c2a_results/baseline_equity.csv"
      "data/c2a_results/baseline_events.csv"
      "data/c2a_results/baseline_trades.csv"
      "data/c2a_results/data_audit.json"
      "data/c2a_results/latest_signal.json"
      "data/c2a_results/latest_state.json"
      "data/c2a_results/latest_training_grid.csv"
      "data/c2a_results/short_window_analysis.json"
      "data/c2a_results/walk_forward_oos_trades.csv"
      "data/c2a_results/walk_forward_selections.csv"
      "reports/c2a_2026_report.md"
    )
  fi
}

record_failure() {
  local reason="$1"
  if [[ -x "$python_bin" ]] && PYTHONPATH=src "$python_bin" -m robot_quant.c2a_cloud \
    failure --project-root . --failed-phase "$phase" --reason "$reason"; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 scripts/record_c2a_failure.py \
      --project-root . --phase "$phase" --date "$trade_day" --reason "$reason"
    return $?
  fi
  echo "C2-A 无可用 Python 解释器，无法写入失败状态" >&2
  return 1
}

stage_phase_results() {
  phase_paths
  staged_paths=()
  local path
  for path in "${paths[@]}"; do
    if [[ -e "$path" ]] || git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
      git add -- "$path" || return 1
      staged_paths+=("$path")
    fi
  done
}

git_network() {
  command -v timeout >/dev/null 2>&1 || {
    echo "缺少 GNU timeout，拒绝无界 Git 网络请求" >&2
    return 127
  }
  GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    GIT_HTTP_LOW_SPEED_LIMIT=1 GIT_HTTP_LOW_SPEED_TIME=5 \
    timeout --signal=TERM --kill-after=2s 8s git "$@"
}

push_with_rebase_retry() {
  local result_commit="$1"
  local attempt
  local ahead_count
  local origin_head
  local parent_commit
  for attempt in 1 2 3; do
    origin_head="$(git rev-parse origin/main)" || return 1
    if git merge-base --is-ancestor "$result_commit" origin/main; then
      return 0
    fi
    ahead_count="$(git rev-list --count origin/main.."$result_commit")" || return 1
    parent_commit="$(git rev-parse "${result_commit}^")" || return 1
    if [[ "$ahead_count" != "1" || "$parent_commit" != "$origin_head" ]]; then
      echo "拒绝推送非单一受控结果提交" >&2
      return 1
    fi
    if git_network push origin "${result_commit}:main"; then
      return 0
    fi
    git_network fetch origin main || continue
    if git merge-base --is-ancestor "$result_commit" origin/main; then
      return 0
    fi
    if [[ "$(git rev-parse HEAD)" != "$result_commit" ]]; then
      echo "阶段运行期间云端 HEAD 发生变化" >&2
      return 1
    fi
    if ! git rebase origin/main; then
      git rebase --abort >/dev/null 2>&1 || true
      return 1
    fi
    result_commit="$(git rev-parse HEAD)" || return 1
  done
  return 1
}

publish_phase_results() {
  local result_commit
  local result_parent
  if [[ -z "$trusted_head" || "$(git rev-parse HEAD)" != "$trusted_head" ]]; then
    echo "阶段运行期间云端 HEAD 发生变化" >&2
    return 1
  fi
  stage_phase_results || return 1
  if [[ "${#staged_paths[@]}" -eq 0 ]] || \
    git diff --cached --quiet -- "${staged_paths[@]}"; then
    echo "C2-A ${phase} 没有结果变化"
    return 0
  fi

  git config user.name "robot-quant-cloud[bot]"
  git config user.email "robot-quant-cloud[bot]@users.noreply.github.com"
  git commit --only -m "chore: update C2-A ${phase} report ${trade_day}" -- \
    "${staged_paths[@]}" || return 1
  result_commit="$(git rev-parse HEAD)" || return 1
  result_parent="$(git rev-parse "${result_commit}^")" || return 1
  if [[ "$result_parent" != "$trusted_head" ]]; then
    echo "拒绝发布非可信基线上的结果提交" >&2
    return 1
  fi
  push_with_rebase_retry "$result_commit"
}

publish_failure_outbox() {
  local reason="$1"
  local origin_url
  local outbox_root
  local outbox_repo
  origin_url="$(git remote get-url origin)" || return 1
  if [[ "$origin_url" == -* || "$origin_url" == *$'\n'* || "$origin_url" == *"://"*"@"* ]]; then
    echo "拒绝使用不安全的 Git 远端" >&2
    return 1
  fi
  outbox_root="$(mktemp -d "/tmp/robot-quant-c2a.XXXXXX")" || return 1
  outbox_repo="${outbox_root}/repo"
  (
    trap 'rm -rf -- "$outbox_root"' EXIT
    git_network clone --quiet --depth 1 --branch main "$origin_url" "$outbox_repo" || exit 1
    command -v python3 >/dev/null 2>&1 || exit 1
    python3 "$failure_recorder" \
      --project-root "$outbox_repo" --phase "$phase" --date "$trade_day" --reason "$reason" \
      || exit 1
    cd "$outbox_repo" || exit 1
    trusted_head="$(git rev-parse HEAD)" || exit 1
    publish_phase_results
  )
}

fail_phase() {
  local reason="$1"
  local exit_code="${2:-1}"
  record_failure "$reason" || true
  publish_phase_results || true
  exit "$exit_code"
}

fail_phase_outbox() {
  local reason="$1"
  local exit_code="${2:-1}"
  publish_failure_outbox "$reason" || true
  exit "$exit_code"
}

lock_acquired=0
handle_termination() {
  trap - TERM INT
  if [[ "$lock_acquired" -eq 1 ]]; then
    fail_phase "C2-A 阶段收到终止信号或达到服务超时" 143
  else
    fail_phase_outbox "C2-A 在等待阶段锁时收到终止信号" 143
  fi
}
trap handle_termination TERM INT

exec 9>"$lock_file"
if ! flock -w 90 9; then
  fail_phase_outbox "云端阶段锁等待超过90秒" 75
fi
lock_acquired=1

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  fail_phase_outbox "云端仓库存在未提交的已跟踪改动" 2
fi

if ! git_network fetch origin main; then
  fail_phase_outbox "云端仓库无法同步 origin/main" 1
fi

ahead_count=""
if ! ahead_count="$(git rev-list --count origin/main..HEAD)"; then
  fail_phase_outbox "云端仓库无法校验本地提交边界" 1
fi
if [[ "$ahead_count" != "0" ]]; then
  fail_phase_outbox "云端仓库存在未推送的本地提交" 2
fi

if ! git rebase origin/main; then
  git rebase --abort >/dev/null 2>&1 || true
  fail_phase_outbox "云端仓库无法同步 origin/main" 1
fi
trusted_head="$(git rev-parse HEAD)" || fail_phase_outbox \
  "云端仓库无法确定可信基线" 1

if [[ ! -x "$python_bin" ]]; then
  fail_phase "云端虚拟环境解释器不可用" 1
fi

phase_exit=0
PYTHONPATH=src "$python_bin" -m robot_quant.c2a_cloud \
  "$phase" --project-root . --service-mode || phase_exit=$?
if [[ "$phase_exit" -ne 0 && "$phase_exit" -ne 75 ]]; then
  fail_phase "C2-A 阶段进程异常退出" 1
fi

publish_phase_results || exit 1
trap - TERM INT
exit "$phase_exit"
