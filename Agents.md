# AGENTS.md — TumbleScope AI Coding Agent Governance

This file governs how AI coding agents (Gemini, Claude, Codex, and similar tools) operate in the `TumbleScope` repository. Agents must implement the agreed design predictably and safely without architectural drift, silent interface changes, or cross-developer conflicts.

---

## 1. Document Hierarchy and Precedence

Four documents govern development, each with a distinct role:

| Document | Role | Authority |
| :--- | :--- | :--- |
| **`CONTRACTS.md`** | Defines **WHAT** must be built | **Authoritative technical contract** (`SCHEMA_VERSION = 1`, frozen at `v0-skeleton`). Governs architecture, module boundaries, function signatures, dataclasses, required/optional keys, error/skip semantics, validation rules, and system invariants. If code and `CONTRACTS.md` disagree, `CONTRACTS.md` wins. |
| **`AGENTS.md`** *(this file)* | Defines **HOW** agents must work | Governs repository inspection, contract compliance, developer ownership enforcement, ambiguity/contradiction handling, testing, Git safety, and reporting. |
| **`TumbleScopeSplit`** | Defines **WHO** owns **WHAT** | Governs module and file ownership between **P1** and **P2**, task blocks, checkpoints (`CP0`–`CP5`), and fallback/drop priorities. Ensure this planning document is committed at a known path in the repository (e.g., `docs/TumbleScopeSplit.md` or root `TumbleScopeSplit.md`) so agents can read it directly. |
| **`README.md`** | Provides **context and setup** | Describes project background, overview, tech stack, and usage context. Distinguish design specifications and estimates from actual measured performance results: any TumbleScope experimental or performance result must be traceable to a file in `docs/results/`. External factual claims must retain their appropriate sources, and unimplemented capabilities must be marked `planned`. |

**Precedence and non-duplication rule:** Do not duplicate technical specifications from `CONTRACTS.md` into `AGENTS.md`, code comments, or new docs—reference `CONTRACTS.md` directly. Within `TumbleScopeSplit`, later addenda (Sections 14 and 16) override earlier sections where they differ, while `CONTRACTS.md` remains the sole authoritative technical interface contract.

---

## 2. Mandatory Agent Behavior

### A. Architecture and Contract Discipline
1. **Read relevant contracts first:** Before implementing or modifying any module, read all sections of `CONTRACTS.md` relevant to your assigned task and the corresponding task block in `TumbleScopeSplit`.
2. **Preserve frozen interfaces:** Never silently change function names, parameter names, parameter defaults, return types, dataclass declarations/fields, constant tuples, required dictionary keys, or error conventions defined in `CONTRACTS.md`. Root-level tests (`tests/test_*.py`) enforce exact signatures via `inspect.signature` automatically.
3. **No independent redesign or scope creep:** Implement the agreed architecture in `CONTRACTS.md` and `TumbleScopeSplit`. Do not introduce unapproved architectural layers, speculative abstractions, duplicate feature pipelines, or features explicitly cut from this build (`TumbleScopeSplit` Section 9; `CONTRACTS.md` Section 5.4).
4. **Contract-change protocol (`CONTRACTS.md` Section 12):**
   - If an existing signature, schema, or contract rule is inconvenient, inefficient, or blocking, **stop and report the issue** to the human developer rather than changing the interface.
   - You may propose an alternative with clear reasoning and downstream consequences, but **wait for explicit human approval** before implementing any contract-level change.
   - Only humans initiate contract changes; only **P2** edits `core/` and `CONTRACTS.md`, bumps `SCHEMA_VERSION`, updates frozen tests, and merges via `contract-change-N`.
5. **Enforce TumbleScope behavioral invariants (`CONTRACTS.md` Sections 0–2, 5–10, 16–21):**
   - **Apparent period only:** Use the apparent (synodic) period in seconds (`period_s`); never call it the sidereal or physical rotation period in code, comments, UI text, or documentation.
   - **Config discipline and `TBD` parameters:** Read configuration from the module's injected `cfg` dictionary. Documented provisional values may be used as specified, but must be read from their designated configuration keys rather than hardcoded in implementation code. `TBD` values must not be invented or used until resolved by their assigned owner (`CONTRACTS.md` Section 21). Follow the single-source period-bounds rule in `CONTRACTS.md` Section 10.
   - **Zero reference-label leakage:** `ReferenceLabel` (`ingestion/reference.py`, `data/reference/`) is strictly evaluation-only and must never be imported or accessed by inference, preprocessing, reconciliation, verification, verdict, pipeline, or UI modules (`CONTRACTS.md` Section 0 & Section 13).
   - **Canonical constants only:** Import reason codes, Stage 2 sub-reasons, skip codes, inconclusive causes, and estimator method names from `core.schemas`; never use free strings or invent new codes (`CONTRACTS.md` Section 2).
   - **Verification and verdict integrity:** If verification was required but is incomplete (`complete=False`), the verdict must never be `Supported` and must follow the `verification_incomplete` rule. If `verification` is `None` because an upstream stage prevented it from running, follow the corresponding verdict rules in `CONTRACTS.md` Sections 5 and 6.
   - **Scientific honesty:** Keep estimator results separate and labelled by source until reconciliation, treat harmonic partners (`P` vs `2P`) as competing hypotheses rather than decoys, and state clearly in UI/docs that a physics fit shows compatibility with a simplified model—never proof of true shape or spin state.

### B. Developer Ownership and Collaboration Boundaries
Respect the strict single-owner file and folder allocation in `CONTRACTS.md` Section 11.1 and `TumbleScopeSplit` Sections 3–6:

- **Person 1 (P1 — Signal, Evaluation, Documentation):**
  - `ingestion/`, `preprocessing/`, `classical_analysis/`, `reconciliation/`, `evaluation/`, `app/plots.py`
  - `configs/preprocess.yaml`, `configs/classical.yaml`, `configs/reconcile.yaml`
  - `data/sdlcd_raw/`, `data/samples/`, `data/reference/`, `data/profiles/sdlcd_profiles.json`
  - `docs/` (excluding P2 result files), `README.md` (after `v0-skeleton`), `tests/p1/`
- **Person 2 (P2 — Models, Physics, Infrastructure, Integration):**
  - `core/`, `physics/`, `ml/`, `verdict/`, `pipeline/`, `app/main.py`, `app/pages/`
  - `configs/default.yaml`, `configs/period_bounds.yaml`, `configs/simulation.yaml`, `configs/cnn.yaml`, `configs/verification.yaml`, `configs/verdict.yaml`, `configs/frozen_thresholds.yaml`
  - `models/`, `scripts/`, `outputs/demo_cache/`, `data/synthetic/`, `data/synthetic_holdout/`, `data/profiles/default_profiles.json`
  - `requirements.txt`, `.gitignore`, `CONTRACTS.md`, `tests/p2/`, `tests/fixtures/*`, and all root-level `tests/test_*.py`
- **`docs/results/` split:** P2 owns `verification_*`, `cnn_training_log.json`, `cnn_calibration.json`, and `runtime_*.json`; P1 owns outputs produced by `evaluation/run_eval.py`.
- **Frozen shared files (P2-only via `CONTRACTS.md` Section 12):** `core/schemas.py`, `core/errors.py`, `core/config.py`, `CONTRACTS.md`, `configs/default.yaml`, `requirements.txt`, `.gitignore`, `tests/fixtures/*`, and root-level `tests/test_*.py`.

**Coordination rules:**
1. Edit **only** files listed in your assigned task block and owned by your developer role.
2. Import shared types only from `core.schemas` (and `IngestionError` from `core.errors`, config helpers from `core.config`). Never import another developer's internal helpers.
3. Never modify another developer's files to work around a defect or missing dependency; stop, identify the affected module/file, and report a minimal reproduction to the human developer.
4. Do not reassign tasks, alter checkpoint deliverables, or edit `requirements.txt` without coordination and P2 approval.

### C. Inspect Before Modifying
1. Check `git status` before starting work. Preserve any pre-existing uncommitted changes and do not overwrite or attribute them to your task.
2. Inspect existing implementations, imports, callers, shared fixtures (`tests/fixtures/`), and tests before making changes.
3. Do not assume a module is empty or broken without inspecting it; Phase 0 provides a working stub pipeline (`is_stub=True`) that satisfies contract tests.
4. Do not overwrite existing implementations or another developer's work.
5. For substantial tasks (new modules, algorithmic changes, or multi-file edits), briefly explain the intended implementation and affected files before proceeding.
6. Keep changes focused strictly on the requested task.

### D. Implementation Discipline
1. Follow the pure-function design and use dataclasses exactly as defined in `core.schemas`. Do not independently change their mutability or declarations.
2. Follow established project technologies (`Python 3.10+`, NumPy, Pandas, SciPy, Astropy, PyTorch, scikit-learn, Streamlit, Plotly, Matplotlib, PyYAML, pytest) and conventions (seeded determinism from config, structured error/skip returns per `CONTRACTS.md` Section 8).
3. Do not introduce new dependencies without concrete justification and P2 approval.
4. Do not replace an existing working implementation with a different approach merely out of personal preference.
5. Clearly distinguish real implementations from stubs (`is_stub=True`), placeholders, approximations, and `TODO`s. Never present an unimplemented feature as functional.
6. Never fabricate execution results, benchmark numbers, test passes, or successful integrations. Distinguish design estimates from measured results in `docs/results/`.
7. Never weaken `.validate()` checks, bypass automated guards, or remove/relax tests to make code pass.

### E. Testing and Integration
1. Run **targeted tests** (`pytest tests/p1 -q` or `pytest tests/p2 -q`) during iterative development, and run the **full test suite (`pytest -q`)** before marking a task complete, at integration points, and at checkpoints.
2. Add or update unit tests in the owner's test folder (`tests/p1/` or `tests/p2/`) when introducing or modifying behavior. Never reuse test filenames across `tests/p1/` and `tests/p2/` (`TumbleScopeSplit` Addendum A6).
3. Verify that modified modules remain compliant with `CONTRACTS.md` (including `.validate()`, JSON round-trip serialization, and signature/guard checks).
4. Test integration seams against shared fixtures (`tests/fixtures/*`) and `pipeline.controller.analyze()` when changes affect multiple modules.
5. Do not claim overall system readiness solely because individual modules pass unit tests. Clearly report any tests that could not be executed and explain why.

### F. Git and Change Control
1. Git authority is separate from file ownership (`CONTRACTS.md` Section 11.2): **only P2 merges into `main`**. Developers work on `p1/dev` and `p2/dev`.
2. **Never autonomously perform destructive or shared Git operations**, including:
   - `git commit` or `git push`
   - `git merge` or `git rebase`
   - `git reset`, `git checkout` (switching branches or discarding changes), or `git clean`
   - Force-pushing, tagging (`v0-skeleton`, `thresholds-frozen`, `v1.0-submission`), or overwriting uncommitted work
   Execute such commands **only** when the human developer explicitly instructs that specific command.
3. If a merge conflict occurs, stop, report the affected files and wait for the human developer's instructions. Do not attempt to resolve or abort the merge independently.
4. Review `git status` and `git diff` before reporting task completion.

---

## 3. Standard Agent Workflow

Apply this 9-step workflow proportionally to the task (trivial edits need only brief checks; substantial tasks require explicit scoping and planning):

1. **Understand:** Read the task prompt and the relevant sections of `CONTRACTS.md` and `TumbleScopeSplit`.
2. **Inspect:** Check `git status` and examine existing module code, imports, callers, fixtures, and tests.
3. **Scope:** Identify target files, affected modules, and P1/P2 ownership boundaries.
4. **Flag:** Report ambiguities, document contradictions, missing dependencies, or needed contract changes **before** proceeding.
5. **Plan:** For substantial tasks, briefly describe the intended implementation and affected files before editing.
6. **Implement:** Make minimal, focused, contract-compliant changes within owned files.
7. **Validate:** Run targeted tests during iteration and the full `pytest -q` suite (plus contract checks) before completion.
8. **Review:** Inspect `git status` and `git diff` for unintended modifications or unrelated edits.
9. **Report:** Summarize changes made, tests run (and any not executed), contract compliance, and unresolved issues or dependencies.

---

## 4. Definition of Done

An agent must not report a task as complete unless all of the following are satisfied:
- [ ] Requested functionality is implemented within the assigned task's file list.
- [ ] All relevant `CONTRACTS.md` interfaces, schemas, invariants, and config rules remain satisfied.
- [ ] No unapproved architecture changes, signature changes, or cross-owner file modifications were made.
- [ ] Relevant targeted tests and the full `pytest -q` suite have been run and pass (or any testing limitations are disclosed).
- [ ] `git status` and `git diff` confirm no pre-existing uncommitted work or unrelated files were overwritten.
- [ ] Remaining limitations, `TODO`s, and cross-developer integration dependencies are clearly reported.