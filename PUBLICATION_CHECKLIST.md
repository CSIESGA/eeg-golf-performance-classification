# GitHub publication checklist

This checklist is for final review before creating a **public** GitHub repository.

- [ ] Confirm the lab's permission applies to this specific code and the aggregate results in `results/recorded_experiments.md`.
- [ ] Read the `.gitignore`; do not use `git add -f` to override it for subject data, checkpoints, or private outputs.
- [ ] Check that no participant IDs, raw EEG recordings, participant-level metadata, authentication tokens, or restricted model weights are present in Git history.
- [ ] If you choose to add a license, verify that you have the rights to license all included source (no license is supplied by default).
- [ ] Confirm the published result table is labeled **historical validation**, not untouched test-set accuracy.
- [ ] Run `python -m compileall -q src scripts` and `pytest -q` after installing dependencies.
- [ ] Run a synthetic smoke test to confirm the CLI works. This does not validate scientific performance.
- [ ] Read the rendered README and check GitHub file links after pushing.

This packaged code was reorganized for GitHub review; do not describe its new module split and synthetic tests as part of the original 2025-2026 laboratory results.
