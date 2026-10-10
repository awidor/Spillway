#!/usr/bin/env bash
# Three-way merge of one Strata pull request into engine/ (our copy of 650cf81): for each file the PR changes,
# base = the merge base of 650cf81 and the PR (NOT 650cf81: a PR from an older commit would otherwise revert every
# later change in its files), ours = engine/<file>, theirs = the PR head.  Conflicts are left marked in the file and listed.
#   bash perfmodel/merge_pr.sh <strata clone> <pr ref> [paths to skip...]
set -u
S=$1; PR=$2; shift 2
E=$(cd "$(dirname "$0")/../engine" && pwd)
T=$(mktemp -d)
B=$(git -C "$S" merge-base 650cf81 "$PR")
rc=0
for f in $(git -C "$S" diff --name-only 650cf81..."$PR"); do
  for skip in "$@"; do [[ $f == $skip* ]] && continue 2; done
  if ! git -C "$S" cat-file -e "$B":"$f" 2>/dev/null; then   # a new file
    mkdir -p "$E/$(dirname "$f")"; git -C "$S" show "$PR":"$f" > "$E/$f"; echo "added    $f"; continue
  fi
  git -C "$S" show "$B":"$f" > "$T/base"; git -C "$S" show "$PR":"$f" > "$T/theirs"
  if git merge-file -L ours -L base -L "$PR" "$E/$f" "$T/base" "$T/theirs"; then echo "merged   $f"
  else echo "CONFLICT $f"; rc=1; fi
done
rm -rf "$T"
exit $rc
