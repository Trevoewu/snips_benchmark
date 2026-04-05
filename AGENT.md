# ML Research Workflow Guidelines

## Philosophy

- A reviewer standard: *"Would a peer reviewer accept this evaluation methodology?"*
- Apply Occam's Razor: if a simpler model achieves 95% of the performance, use it.
- Never claim "better" based on training loss alone.

## Before You Train

- Define upfront: **hypothesis -> baseline -> dataset -> evaluation metric**.
- Establish a strong, simple baseline before building anything complex.
- Audit data for leaks, distribution shifts, and preprocessing bugs before touching model code.
- Check the literature; verify you aren't reinventing the wheel.

## While Training

- If loss diverges (NaN) or severe overfitting occurs, **stop and diagnose**; don't blindly restart.
- Autonomously debug by inspecting: gradient norms, weight initializations, data distributions, and stack traces.
- Resolve OOM errors and shape mismatches by reading the error, not guessing.
- Log intermediate metrics via W&B or TensorBoard.

## Before You Conclude

- Run baseline comparisons and compute statistical significance on held-out test sets.
- Perform ablation studies to isolate which component caused the improvement.
- Scrutinize OOD performance and double-check for test-set leakage.

## After Each Run

- Update `research/lab_notebook.md`: what changed, why, and what the outcome was.
- Track failure modes explicitly (mode collapse, vanishing gradients, OOM) to build domain intuition.
- Review past failures at the start of each new session before planning the next experiment.

## Reproducibility Checklist

- [ ] Random seeds fixed
- [ ] Library versions pinned
- [ ] Data splits identical across all runs
- [ ] Experiment spec written to `experiments/todo.md` before launching
- [ ] Results logged to `experiments/results.md` with charts
