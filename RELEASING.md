# Releasing

Releases are tag-driven: `.github/workflows/release.yml` builds every wheel and
publishes to PyPI (`toktok-rs`) and crates.io (`toktok-rs`).

```sh
scripts/release.sh patch        # or: minor | major | 0.4.2
```

That bumps the version, runs the tests, dry-runs the crate package, commits,
tags and pushes — the tag is what triggers the workflow.

| argument | what it does |
|---|---|
| `patch` / `minor` / `major` | bump that part of the current version (`0.1.0` → `0.1.1` / `0.2.0` / `1.0.0`) |
| `X.Y.Z` | set an exact version, e.g. `0.4.2` or `1.0.0-rc.1` |
| `--dry-run` | do everything — bump, verify, test, package — then revert and print what it *would* have run. Commits nothing, tags nothing, pushes nothing. |
| `--skip-tests` | skip `cargo test`, `pytest` and the `cargo publish` dry run. The release workflow still runs them before it publishes, so this trades local certainty for a faster tag. |
| `-h` / `--help` | usage |

Before touching anything it checks that you are on `main`, the tree is clean,
local and `origin/main` agree, and the tag does not already exist locally or on
the remote. If a later step fails, the version bump is rolled back so you are
never left with a half-bumped checkout.

Doing it by hand is the same thing: bump `[workspace.package] version` in
`Cargo.toml`, then

```sh
git tag v0.1.0 && git push origin v0.1.0
```

The workflow refuses to publish if the tag and the crate version disagree, if
tests fail on that exact commit, or if `cargo publish --dry-run` fails.

To rehearse without publishing anything: run the workflow manually
(**Actions → Release → Run workflow**) with `dry_run` left on. It builds and
checks everything and skips both publish steps.

## One-time setup

### crates.io — publishing rights belong to the repo, not to a person

A crates.io account is tied to a GitHub *user*, so the first publish has to come
from a person's token. After that, ownership and publishing rights move to the
organisation and nobody's personal token is needed again.

1. **First publish, from a laptop** (crates.io has no crate to attach a trusted
   publisher to until it exists):

   ```sh
   cargo login          # paste a token from https://crates.io/settings/tokens
   cargo publish -p toktok-rs
   ```

2. **Hand ownership to the org.** GitHub teams can own crates, so publishing
   rights follow team membership rather than one person:

   ```sh
   cargo owner --add github:vectorize-io:<team-slug> toktok-rs
   ```

   The team must exist in the vectorize-io org and you must be a member of it.
   The crate keeps you as an owner too; that is fine, and you can step down
   later with `cargo owner --remove <your-username> toktok-rs` once someone else
   or the team is in place.

3. **Turn on Trusted Publishing** so CI never needs a token again — on
   <https://crates.io/crates/toktok-rs/settings> add a trusted publisher:

   | field | value |
   |---|---|
   | Repository owner | `vectorize-io` |
   | Repository name | `toktok` |
   | Workflow filename | `release.yml` |
   | Environment | `crates-io` |

   From then on `rust-lang/crates-io-auth-action` in the release workflow mints a
   short-lived token scoped to this repository. **This is the answer to "my token
   is bound to my personal account"** — after step 3 the token is irrelevant, the
   *repository* is what crates.io trusts.

Until step 3 is done, the workflow falls back to a `CARGO_REGISTRY_TOKEN` secret
in the `crates-io` environment, so a personal token in repo secrets also works.

### PyPI

Same idea, and PyPI supports it for brand-new projects, so no token is ever
needed. At <https://pypi.org/manage/account/publishing/> add a **pending**
trusted publisher:

| field | value |
|---|---|
| PyPI project name | `toktok-rs` |
| Owner | `vectorize-io` |
| Repository name | `toktok` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Then create the `pypi` and `crates-io` environments under
**Settings → Environments** in the repo. Adding required reviewers to them is
worth it: a release then waits for a human to approve before it publishes.

## What gets published

| artifact | registry | covers |
|---|---|---|
| `toktok_rs-<v>-cp311-abi3-*.whl` | PyPI | CPython 3.11–3.14+, GIL builds, one wheel per platform |
| `toktok_rs-<v>-cp314-cp314t-*.whl` | PyPI | free-threaded CPython 3.14t (no stable ABI exists, so this one is version-specific) |
| `toktok_rs-<v>.tar.gz` | PyPI | sdist, builds from source with a Rust toolchain |
| `toktok-rs <v>` | crates.io | the Rust crate, with the vocabularies embedded |

Platforms: Linux x86_64 and aarch64 (manylinux), macOS universal2, Windows x64.

## Notes

- **crates.io publishes are permanent.** A version can be yanked but never
  replaced or deleted, so let the dry run pass first.
- The crate and the PyPI distribution are both `toktok-rs`, and both import as
  `toktok` (`use toktok::…` / `import toktok`). Plain `toktok` was taken on both
  registries by unrelated projects.
- Bump the version in `[workspace.package]` in the root `Cargo.toml`; the Python
  package takes its version from the crate.
