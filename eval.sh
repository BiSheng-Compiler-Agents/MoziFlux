#!/bin/bash
# eval.sh — KernelGen integration tool
#
# Subcommands:
#   clone                       — clone KernelGen into $HOME/KernelGen
#   push [remote] [branch]      — push a branch to origin with creds
#                                 (uses -u, works for first-time pushes)
#   status                      — show KernelGen remotes + last commit
#   submit [--kernel NAME]      — iterate over KernelBench_Triton; for each
#                                 kernel dir that has all 5 deliverables
#                                 present, push it to a fresh
#                                 `kernel/<name>` branch in KernelGen. Wait
#                                 for CI to populate results.txt, pull the
#                                 branch, and copy results.txt back to the
#                                 original kernel dir.
#
# Authentication
# --------------
# For a *public* clone (the default), no credentials are required.
# For a *private* clone or any push/submit operation, set these two env
# vars before invoking the script:
#
#     export GITHUB_USERNAME="your-name"
#     export GITHUB_TOKEN=***        # classic PAT or fine-grained token
#
# The script embeds them in the URL only at the moment of use, so the
# script itself stays safe to commit. NEVER hardcode a token in this
# file.
#
# Tunables (env vars)
# -------------------
#   GITHUB_USERNAME, GITHUB_TOKEN       Required for push/submit/private clone
#   CI_MAX_WAIT_SECONDS=1800            Give up polling for results.txt after N s
#   CI_POLL_INTERVAL_SECONDS=10         Seconds between remote-refresh polls
#   KERNELGEN_BRANCH_BASE=main          Base branch to fork kernel branches from
#   KERNELBENCH_DIR=...                 Override the KernelBench_Triton path

set -euo pipefail
IFS=$'\n\t'

# ── config ───────────────────────────────────────────────────────────────────

KERNELGEN_DIR="${KERNELGEN_DIR:-$HOME/KernelGen}"
# Override the target repo with `KERNELGEN_REPO=your-user/KernelGen bash eval.sh ...`
# to push to a fork instead of the upstream. The script embeds this in the
# auth-bearing URL and the public clone URL.
KERNELGEN_REPO="${KERNELGEN_REPO:-Huawei-CPLLab/KernelGen}"
KERNELGEN_URL_PUBLIC="https://github.com/${KERNELGEN_REPO}.git"
KERNELGEN_BRANCH_BASE="${KERNELGEN_BRANCH_BASE:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELBENCH_DIR="${KERNELBENCH_DIR:-${SCRIPT_DIR}/datasets/KernelBench_Triton}"

CI_MAX_WAIT_SECONDS="${CI_MAX_WAIT_SECONDS:-1800}"
CI_POLL_INTERVAL_SECONDS="${CI_POLL_INTERVAL_SECONDS:-10}"

# Required deliverables — all 5 must exist for a kernel to be submitted.
# This is the same checklist the triton-operator orchestration skill uses.
REQUIRED_DELIVERABLES=(
    "opt_*.py"
    "profile_kernels.py"
    "Optimizations.md"
    "performance_report.md"
    "review.md"
)

# ── helpers ──────────────────────────────────────────────────────────────────

# Print an auth-bearing URL or fail with a clear error.
# Format: https://<username>:<token>@github.com/<owner>/<repo>.git
_kgen_url_with_creds() {
    if [ -z "${GITHUB_USERNAME:-}" ] || [ -z "${GITHUB_TOKEN:-}" ]; then
        echo "ERROR: GITHUB_USERNAME and GITHUB_TOKEN must both be set for this operation." >&2
        echo "       export GITHUB_USERNAME=...   export GITHUB_TOKEN=***" >&2
        return 1
    fi
    # 3 %s slots, 3 args. Note: the token is embedded in the URL only at
    # the moment of use; never persisted to disk or echoed to a log.
    printf 'https://%s:%s@github.com/%s.git\n' \
        "$GITHUB_USERNAME" "$GITHUB_TOKEN" "$KERNELGEN_REPO"
}

_kgen_clone() {
    if [ -n "${GITHUB_USERNAME:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
        echo "Cloning with provided GITHUB_USERNAME / GITHUB_TOKEN..."
        git clone "$(_kgen_url_with_creds)" "$KERNELGEN_DIR"
    else
        echo "Cloning public repo -- no creds provided"
        git clone "$KERNELGEN_URL_PUBLIC" "$KERNELGEN_DIR"
    fi
}

_kgen_status() {
    if [ ! -d "$KERNELGEN_DIR/.git" ]; then
        echo "ERROR: $KERNELGEN_DIR is not a git repo." >&2
        return 1
    fi
    git -C "$KERNELGEN_DIR" remote -v
    echo "---"
    git -C "$KERNELGEN_DIR" log -1 --oneline
}

# Push a branch to KernelGen using creds. `-u` works for first-time pushes
# (sets upstream) and is a no-op for already-tracked branches.
#
# Args:
#   $1  branch name on the remote (default: HEAD → current local branch)
#
# We do NOT pass a separate remote NAME (e.g. "origin") to git push.
# The credentialed URL is already the remote specifier; adding "origin"
# as an extra arg confuses git into treating it as a refspec, which
# triggers "fatal: refs/remotes/origin/HEAD cannot be resolved to branch".
_kgen_push_branch() {
    local branch="${1:-HEAD}"
    if [ ! -d "$KERNELGEN_DIR/.git" ]; then
        echo "ERROR: $KERNELGEN_DIR is not a git repo." >&2
        return 1
    fi
    echo "  Pushing $branch ..."
    local creds_url
    creds_url=$(_kgen_url_with_creds) || return 1

    # Clear the local tracking ref so a plain push doesn't get rejected
    # with "stale info" because the tracking ref still points to a
    # commit we just deleted from the remote. The tracking ref is just
    # local bookkeeping — git will recreate it on the next fetch.
    git -C "$KERNELGEN_DIR" update-ref -d "refs/remotes/origin/${branch}" 2>/dev/null || true

    # Try a PLAIN push first. This works when:
    #   - The remote was just deleted in _submit_one's "delete stale
    #     remote branch" step (push creates a fresh branch)
    #   - The local is ahead of the remote (fast-forward)
    #   - The remote doesn't have the branch yet (first push)
    if git -C "$KERNELGEN_DIR" push -u "$creds_url" "$branch" 2>/dev/null; then
        echo "  Pushed."
        return 0
    fi

    # Plain push failed. Most likely cause: the remote still has commits
    # we don't have (e.g., CI's [skip ci] results.txt commit from a
    # previous run that we couldn't delete — no permission, or the
    # delete step failed silently). Fall back to --force, which is
    # safe for these branches because the user owns them and a fresh
    # CI run on the new code is the desired behavior.
    echo "  plain push failed (remote diverged), retrying with --force..."
    git -C "$KERNELGEN_DIR" push --force -u "$creds_url" "$branch"
    echo "  Pushed."
}

# Poll the remote branch until results.txt's blob hash changes from the
# baseline (or appears, if it didn't exist in our push). Returns 0 on
# detection, 1 on timeout.
#
# Args:
#   $1  branch name (e.g. "kernel/l1_19_ReLU")
#   $2  kernel dir name (e.g. "l1_19_ReLU") — path inside KernelGen
#
# We use blob hashes rather than file content comparison because the
# kernel dir is committed inside KernelGen (not a separate repo), so the
# remote's blob hash for <name>/results.txt is the canonical "what CI
# wrote" signal. Comparing hashes is O(1) and doesn't need to read
# potentially-large perf tables.
_kgen_wait_for_results() {
    local branch="$1"
    local name="$2"
    local rel_path="${name}/results.txt"
    local max_wait="${CI_MAX_WAIT_SECONDS:-1800}"
    local interval="${CI_POLL_INTERVAL_SECONDS:-10}"

    # Baseline = the blob hash in HEAD (our just-pushed commit). If
    # results.txt is not tracked in HEAD, use the sentinel "<absent>".
    local baseline
    baseline=$(git -C "$KERNELGEN_DIR" ls-tree HEAD -- "$rel_path" 2>/dev/null \
                   | awk '{print $3}')
    [ -z "$baseline" ] && baseline="<absent>"

    echo "  Polling for $rel_path to change (max ${max_wait}s, every ${interval}s, baseline=$baseline)..."

    local waited=0
    local current=""
    local fetch_err_shown=0
    # Build the creds-bearing fetch URL once. We can't use the bare remote
    # name (e.g. `git fetch origin <branch>`) because the remote URL
    # requires authentication and there's no credential helper configured
    # in the local repo. Passing the creds in the URL works, but the
    # shorthand `<refspec>` form would create a LOCAL branch ref
    # (refs/heads/<branch>), not the remote-tracking ref we want. So we
    # use the explicit `<src>:<dst>` form — fully-qualified with
    # `refs/heads/...` to avoid any parsing ambiguity from the slashes
    # in the branch name.
    local fetch_url
    fetch_url=$(_kgen_url_with_creds) || return 1
    # `+` prefix forces the ref update so we always pick up CI's commit
    # even when the local tracking ref is stale (after a previous
    # force-push rewrote the remote history, the local tracking ref
    # points to a commit that isn't an ancestor of the new one, and
    # without `+` git refuses with "non-fast-forward").
    local fetch_refspec="+refs/heads/${branch}:refs/remotes/origin/${branch}"
    while [ "$waited" -lt "$max_wait" ]; do
        sleep "$interval"
        waited=$((waited + interval))

        # Refresh the remote-tracking ref. Capture the error so we can
        # surface it (once) if the fetch keeps failing — otherwise we'd
        # silently spin for 30 minutes.
        local fetch_err=""
        fetch_err=$(git -C "$KERNELGEN_DIR" fetch "$fetch_url" "$fetch_refspec" 2>&1) \
            && fetch_err=""
        if [ -n "$fetch_err" ]; then
            if [ "$fetch_err_shown" -eq 0 ]; then
                echo
                echo "  fetch error (will keep retrying, suppressing further): $fetch_err" >&2
                fetch_err_shown=1
            fi
            printf '.' >&2
            continue
        fi

        # Get the current blob hash at the same path in the remote branch.
        current=$(git -C "$KERNELGEN_DIR" ls-tree "origin/${branch}" -- "$rel_path" 2>/dev/null \
                       | awk '{print $3}')
        [ -z "$current" ] && current="<absent>"

        if [ "$current" != "$baseline" ]; then
            echo
            echo "  ✓ $rel_path changed after ${waited}s ($baseline → $current)"
            return 0
        fi
        printf '.' >&2
    done
    echo
    echo "  WARNING: $rel_path did not change within ${max_wait}s (still $current, baseline was $baseline)"
    return 1
}

# Returns 0 if all 5 deliverables are present in $1, 1 otherwise.
_has_all_deliverables() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    # opt_*.py — at least one file matching the glob
    compgen -G "${dir}/opt_*.py" > /dev/null || return 1
    [ -f "${dir}/profile_kernels.py" ]    || return 1
    [ -f "${dir}/Optimizations.md" ]      || return 1
    [ -f "${dir}/performance_report.md" ] || return 1
    [ -f "${dir}/review.md" ]             || return 1
    return 0
}

# Copy top-level files of a kernel dir into a destination in KernelGen.
# No subdirectories are copied (per the user's spec).
_copy_kernel_files() {
    local src="$1"   # .../KernelBench_Triton/<name>
    local dst="$2"   # $KERNELGEN_DIR/<name>
    rm "$src/results.txt"
    rm -rf "$dst"
    mkdir -p "$dst"
    find "$src" -maxdepth 1 -type f -exec cp -t "$dst" {} +
}

# Write the auto-generated eval.sh into the KernelGen copy of the kernel.
#
# CI contract (see .github/workflows/ci.yml in KernelGen):
#   1. The runner looks for an executable `eval.sh` (or `eval.py`) in the
#      single subdirectory that was added/modified in the push. We MUST
#      chmod +x it (done below).
#   2. The runner sources `compilerclaw_setenv.sh` on the host first, so we
#      do NOT need to source CANN env ourselves.
#   3. The runner invokes us as `../<EVAL_DIR>/eval.sh >> ./results.txt`
#      from inside the kernel dir. So:
#        - $0 is a *relative* path (../<name>/eval.sh), NOT a path that
#          cd "$(dirname "$0")" would resolve correctly from the cwd
#          (which is the kernel dir, not its parent). We must resolve
#          $0 to an absolute path with `readlink -f` first.
#        - The CI's `>> ./results.txt` already captures our stdout into
#          results.txt, but ONLY stdout — stderr from triton import
#          failures or numerical mismatches would be lost. We open our
#          own append handle (`exec >> ./results.txt 2>&1`) so stderr
#          also lands in the file. Both handles write to the same file
#          with O_APPEND, which is atomic per write — no interleaving.
#        - The CI uses `>>` (append) not `>` (truncate), so on re-pushes
#          new measurements get appended to existing results.txt.
#   4. We run BOTH --test and --bench so results.txt contains the
#      correctness check (catches numerical drift after code changes)
#      AND the perf table. profile_kernels.py runs unit_test() first
#      then benchmark.run(...) when both flags are present.
_create_kernel_eval_sh() {
    local kernel_dir="$1"
    cat > "${kernel_dir}/eval.sh" <<'EOF'
#!/bin/bash
# Auto-generated by CompilerClaw/eval.sh (submit subcommand).
# Runs correctness check + benchmark, capturing stdout AND stderr
# into ./results.txt in this directory.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
python profile_kernels.py --test --bench
EOF
    chmod +x "${kernel_dir}/eval.sh"
}

# Process a single kernel dir end-to-end:
#   1. validate deliverables
#   2. make sure base branch is current in KernelGen
#   3. create a fresh `kernel/<name>` branch
#   4. copy the kernel's top-level files into it
#   5. write a per-kernel eval.sh (CI's entrypoint)
#   6. commit and push
#   7. wait for CI to populate results.txt
#   8. pull the branch
#   9. copy results.txt back to the source kernel dir
_submit_one() {
    local src_dir="$1"
    local name
    name="$(basename "$src_dir")"
    local branch="kernel/${name}"
    local dst_dir="${KERNELGEN_DIR}/${name}"

    echo ""
    echo "══ Submitting ${name} → ${branch} ══"

    if ! _has_all_deliverables "$src_dir"; then
        echo "  SKIP: missing one or more of: ${REQUIRED_DELIVERABLES[*]}"
        return 0
    fi

    if [ ! -d "$KERNELGEN_DIR/.git" ]; then
        echo "  ERROR: $KERNELGEN_DIR is not a git repo."
        echo "         Run 'bash eval.sh clone' first."
        return 1
    fi

    pushd "$KERNELGEN_DIR" > /dev/null

    # Make sure base branch is checked out and up to date.
    if ! git checkout "$KERNELGEN_BRANCH_BASE" 2>/dev/null; then
        echo "  ERROR: base branch '$KERNELGEN_BRANCH_BASE' not found locally"
        popd > /dev/null
        return 1
    fi
    if ! git -C "$KERNELGEN_DIR" pull --ff-only "$(_kgen_url_with_creds)" "$KERNELGEN_BRANCH_BASE" 2>/dev/null; then
        echo "  WARNING: pull of $KERNELGEN_BRANCH_BASE failed (continuing with local state)"
    fi

    # Reuse a local branch only if it's already merged into the base.
    # If it has unmerged commits, refuse — don't destroy user work.
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        if git merge-base --is-ancestor "$branch" "$KERNELGEN_BRANCH_BASE"; then
            echo "  Deleting already-merged local branch $branch"
            git branch -D "$branch"
        else
            echo "  ERROR: local branch $branch has unmerged commits; refusing to delete."
            echo "         Resolve manually: cd $KERNELGEN_DIR && git branch -D $branch"
            popd > /dev/null
            return 1
        fi
    fi

    # Best-effort delete of a stale remote branch so the new push creates
    # fresh history. Tolerate failure (might not exist, or no perms).
    # Note: ls-remote needs creds too (private repo) — pass the creds URL.
    if git ls-remote --heads "$(_kgen_url_with_creds)" "$branch" 2>/dev/null | grep -q "$branch"; then
        echo "  Deleting stale remote branch $branch"
        if ! git push "$(_kgen_url_with_creds)" --delete "$branch" 2>/dev/null; then
            echo "  WARNING: could not delete remote $branch (continuing)"
        fi
    fi

    # Create the fresh branch off the base.
    git checkout -b "$branch" "$KERNELGEN_BRANCH_BASE"
    popd > /dev/null

    # Copy the kernel's files and generate the per-kernel eval.sh.
    _copy_kernel_files "$src_dir" "$dst_dir"
    _create_kernel_eval_sh "$dst_dir"

    # Commit and push.
    pushd "$KERNELGEN_DIR" > /dev/null
    git add "$name"
    if git diff --cached --quiet; then
        echo "  Nothing to commit (kernel already up to date on this branch)"
    else
        git -c user.name="eval.sh" -c user.email="eval@local" \
            commit -m "Add kernel ${name}"
    fi
    _kgen_push_branch "$branch"

    # Poll the remote until CI commits an updated results.txt, then pull.
    # The poll detects the change as soon as CI pushes its [skip ci] commit
    # back to the branch — no fixed sleep required.
    _kgen_wait_for_results "$branch" "$name" || true
    echo "  Pulling $branch ..."
    if ! git -C "$KERNELGEN_DIR" pull --ff-only "$(_kgen_url_with_creds)" "$branch" 2>/dev/null; then
        echo "  WARNING: pull failed (CI may not have committed yet)"
    fi
    popd > /dev/null

    # Copy results.txt back to the source kernel dir.
    if [ -f "${dst_dir}/results.txt" ]; then
        cp "${dst_dir}/results.txt" "${src_dir}/results.txt"
        echo "  ✓ Copied results.txt → ${src_dir}/results.txt"
    else
        echo "  WARNING: ${dst_dir}/results.txt not found after wait"
    fi
}

# Iterate over all kernel dirs and submit each that has the deliverables.
# If $1 is set, only process that one kernel.
_submit_all() {
    local only_kernel="${1:-}"
    if [ ! -d "$KERNELBENCH_DIR" ]; then
        echo "ERROR: KernelBench_Triton not found at $KERNELBENCH_DIR" >&2
        return 1
    fi
    local count=0
    local submitted=0
    local skipped=0
    local failed=0
    for dir in "${KERNELBENCH_DIR}"/*/; do
        [ -d "$dir" ] || continue
        local name
        name="$(basename "$dir")"
        if [ -n "$only_kernel" ] && [ "$name" != "$only_kernel" ]; then
            continue
        fi
        count=$((count + 1))
        if _submit_one "$dir"; then
            if _has_all_deliverables "$dir"; then
                submitted=$((submitted + 1))
            else
                skipped=$((skipped + 1))
            fi
        else
            failed=$((failed + 1))
        fi
    done
    echo ""
    echo "══ Done. Inspected ${count} kernel(s): ${submitted} submitted, ${skipped} skipped, ${failed} failed. ══"
}

# ── main ─────────────────────────────────────────────────────────────────────

cmd="${1:-clone}"
shift || true

case "$cmd" in
    clone|"")
        if [ -d "$KERNELGEN_DIR" ]; then
            echo "KernelGen directory exists at $KERNELGEN_DIR — skipping clone."
        else
            _kgen_clone
            echo "Cloned KernelGen into $KERNELGEN_DIR"
        fi
        ;;
    push)
        _kgen_push_branch "${1:-HEAD}"
        ;;
    status)
        _kgen_status
        ;;
    submit)
        only=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --kernel) only="${2:-}"; shift 2 ;;
                *) echo "ERROR: unknown submit arg: $1" >&2; exit 2 ;;
            esac
        done
        _submit_all "$only"
        ;;
    *)
        cat >&2 <<'EOF'
Usage: bash eval.sh [COMMAND] [ARGS]

Commands:
  clone                       Clone KernelGen into $HOME/KernelGen if missing
  push [remote] [branch]      Push a branch with creds (uses -u; first-time
                              pushes set upstream). Defaults: origin, HEAD.
  status                      Show KernelGen remotes + last commit
  submit [--kernel NAME]      Iterate over KernelBench_Triton; for each kernel
                              with all 5 deliverables present, push to a fresh
                              'kernel/<name>' branch in KernelGen, poll for CI to
                              commit results.txt, pull, and copy results.txt back.

Tunables (env vars):
  GITHUB_USERNAME, GITHUB_TOKEN       Required for push / submit / private clone
  CI_MAX_WAIT_SECONDS=1800            Give up polling for results.txt after N s
  CI_POLL_INTERVAL_SECONDS=10         Seconds between remote-refresh polls
  KERNELGEN_BRANCH_BASE=main          Base branch to fork kernel branches from
  KERNELBENCH_DIR=...                 Override the KernelBench_Triton path
EOF
        exit 2
        ;;
esac
