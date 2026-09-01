#!/usr/bin/env bash
#
# Cut a release: bump the version, commit, tag, push. CI does the rest —
# pushing the tag triggers .github/workflows/release.yml, which builds every
# wheel and publishes to PyPI and crates.io (see RELEASING.md).
#
#   scripts/release.sh patch          # 0.1.0 -> 0.1.1
#   scripts/release.sh minor          # 0.1.0 -> 0.2.0
#   scripts/release.sh major          # 0.1.0 -> 1.0.0
#   scripts/release.sh 0.4.2          # an exact version
#   scripts/release.sh patch --dry-run
#
# The version lives in one place: [workspace.package] version in Cargo.toml.
# Both crates inherit it, and the Python package takes its version from the
# bindings crate (pyproject declares version as dynamic), so there is nothing
# to keep in sync by hand — this script verifies that rather than trusting it.

set -euo pipefail

cd "$(dirname "$0")/.."

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
die() { echo "${RED}error:${OFF} $*" >&2; exit 1; }
step() { echo "${GREEN}==>${OFF} $*"; }
note() { echo "    ${DIM}$*${OFF}"; }

DRY_RUN=0
SKIP_TESTS=0
BUMP=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option $arg" ;;
    *) [ -z "$BUMP" ] || die "give one version or bump level, not '$BUMP' and '$arg'"; BUMP="$arg" ;;
  esac
done
[ -n "$BUMP" ] || die "usage: scripts/release.sh {major|minor|patch|X.Y.Z} [--dry-run] [--skip-tests]"

command -v cargo >/dev/null || die "cargo not found"
command -v uv >/dev/null || die "uv not found (https://docs.astral.sh/uv/)"

# ---------------------------------------------------------------- current state
CURRENT=$(cargo metadata --format-version 1 --no-deps --offline 2>/dev/null \
  | python3 -c "import json,sys;print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='toktok-rs'))")
[ -n "$CURRENT" ] || die "could not read the current version from Cargo.toml"

case "$BUMP" in
  major|minor|patch)
    NEW=$(python3 - "$CURRENT" "$BUMP" <<'PY'
import sys
major, minor, patch = (int(x) for x in sys.argv[1].split("."))
level = sys.argv[2]
if level == "major":
    major, minor, patch = major + 1, 0, 0
elif level == "minor":
    minor, patch = minor + 1, 0
else:
    patch += 1
print(f"{major}.{minor}.{patch}")
PY
) ;;
  *)
    [[ "$BUMP" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]] \
      || die "'$BUMP' is not a semver version or one of major|minor|patch"
    NEW="$BUMP" ;;
esac

TAG="v$NEW"
step "Releasing ${YELLOW}$CURRENT${OFF} -> ${YELLOW}$NEW${OFF}  (tag $TAG)"
[ "$DRY_RUN" = 1 ] && note "dry run: nothing will be committed, tagged or pushed"

# ------------------------------------------------------------------ safety net
step "Checking the working tree"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "main" ] || die "on branch '$BRANCH'; release from main"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty; commit or stash first"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || die "local main and origin/main have diverged; pull or push first"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
  && die "tag $TAG already exists"
if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
  die "tag $TAG already exists on origin"
fi
note "on main, clean, in sync with origin, $TAG is free"

# -------------------------------------------------------------------- the bump
# From here on the tree carries an uncommitted bump. Undo it on any failure so a
# botched release never leaves a half-bumped checkout behind.
BUMP_APPLIED=0
restore_on_failure() {
  local code=$?
  if [ "$code" != 0 ] && [ "$BUMP_APPLIED" = 1 ]; then
    echo "${YELLOW}reverting the version bump${OFF}" >&2
    git checkout -- Cargo.toml Cargo.lock
  fi
  exit "$code"
}
trap restore_on_failure EXIT

step "Bumping [workspace.package] version in Cargo.toml"
python3 - "$CURRENT" "$NEW" <<'PY'
import pathlib, re, sys
current, new = sys.argv[1], sys.argv[2]
p = pathlib.Path("Cargo.toml")
s = p.read_text()
# only the version inside [workspace.package], not any dependency's
new_text, n = re.subn(
    r'(\[workspace\.package\][^\[]*?\bversion\s*=\s*")%s(")' % re.escape(current),
    r"\g<1>%s\g<2>" % new,
    s,
    flags=re.S,
)
if n != 1:
    sys.exit(f"expected exactly one [workspace.package] version = \"{current}\", found {n}")
p.write_text(new_text)
PY
BUMP_APPLIED=1
cargo update -q -w  # refresh Cargo.lock with the new version
note "Cargo.toml + Cargo.lock updated"

step "Verifying every artifact reports $NEW"
CRATE_V=$(cargo metadata --format-version 1 --no-deps --offline \
  | python3 -c "import json,sys;print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='toktok-rs'))")
[ "$CRATE_V" = "$NEW" ] || die "crate reports $CRATE_V, expected $NEW"
# the wheel's version is dynamic — it comes from the bindings crate, so check it
WHEEL_V=$(cargo metadata --format-version 1 --no-deps --offline \
  | python3 -c "import json,sys;print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='toktok-py'))")
[ "$WHEEL_V" = "$NEW" ] || die "python bindings crate reports $WHEEL_V, expected $NEW"
note "crate $CRATE_V · wheel $WHEEL_V"

# ----------------------------------------------------------------- the checks
if [ "$SKIP_TESTS" = 1 ]; then
  echo "${YELLOW}warning:${OFF} skipping tests — CI will still run them before publishing"
else
  step "Running the tests (the release workflow reruns these before it publishes)"
  cargo test --release -q
  uv sync -q --no-editable
  uv run -q pytest -q
  step "Dry-running the crate package"
  # --allow-dirty: the working tree was verified clean above, so the only
  # uncommitted change is the version bump this script just made
  cargo publish --dry-run -q --allow-dirty -p toktok-rs
fi

# ------------------------------------------------------------ commit, tag, push
if [ "$DRY_RUN" = 1 ]; then
  step "Dry run complete — reverting the version bump"
  git checkout -- Cargo.toml Cargo.lock
  BUMP_APPLIED=0
  echo
  echo "Would have run:"
  echo "  git commit -am 'Release $NEW' && git tag -a $TAG && git push origin main $TAG"
  exit 0
fi

step "Committing and tagging"
git commit -q -am "Release $NEW"
BUMP_APPLIED=0   # committed: nothing left to revert
git tag -a "$TAG" -m "toktok $NEW"

step "Pushing to origin"
git push -q origin main
git push -q origin "$TAG"

echo
echo "${GREEN}Released $NEW.${OFF}"
echo "  The release workflow is now building wheels and publishing:"
echo "    https://github.com/vectorize-io/toktok/actions/workflows/release.yml"
echo "  When it finishes:"
echo "    pip install toktok-rs==$NEW"
echo "    cargo add toktok-rs@$NEW"
echo
echo "  ${DIM}First release? PyPI and crates.io need their one-time setup first —"
echo "  see RELEASING.md. The workflow will fail at the publish step without it.${OFF}"
