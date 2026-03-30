1 # ML Research Workflow Guidelines
  2
  3 ## Philosophy
  4 - A reviewer standard: *"Would a peer reviewer accept this evaluation methodology?"*
  5 - Apply Occam's Razor: if a simpler model achieves 95% of the performance, use it.
  6 - Never claim "better" based on training loss alone.
  7
  8 ## Before You Train
  9 - Define upfront: **hypothesis → baseline → dataset → evaluation metric**.
 10 - Establish a strong, simple baseline before building anything complex.
 11 - Audit data for leaks, distribution shifts, and preprocessing bugs before touching model code.
 12 - Check the literature — verify you aren't reinventing the wheel.
 13
 14 ## While Training
 15 - If loss diverges (NaN) or severe overfitting occurs, **stop and diagnose** — don't blindly restart.
 16 - Autonomously debug by inspecting: gradient norms, weight initializations, data distributions, and stack traces.
 17 - Resolve OOM errors and shape mismatches by reading the error, not guessing.
 18 - Log intermediate metrics via W&B or TensorBoard.
 19
 20 ## Before You Conclude
 21 - Run baseline comparisons and compute statistical significance on held-out test sets.
 22 - Perform ablation studies to isolate which component caused the improvement.
 23 - Scrutinize OOD performance and double-check for test-set leakage.
 24
 25 ## After Each Run
 26 - Update `research/lab_notebook.md`: what changed, why, and what the outcome was.
 27 - Track failure modes explicitly (mode collapse, vanishing gradients, OOM) to build domain intuition.
 28 - Review past failures at the start of each new session before planning the next experiment.
 29
 30 ## Reproducibility Checklist
 31 - [ ] Random seeds fixed
 32 - [ ] Library versions pinned
 33 - [ ] Data splits identical across all runs
 34 - [ ] Experiment spec written to `experiments/todo.md` before launching
 35 - [ ] Results logged to `experiments/results.md` with charts
