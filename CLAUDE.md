Manually test every change you make by running the appropriate script or CLI command. When you run the script, frequently monitor the output until it appears to be running without issue, and then check again every 30 seconds until either 3 minutes have passed or multiple iteration loops of the main computation have run without error. If you find an error unrelated to your task, at minimum quote back the exact error to the user after completing your task.

# Experiment Logs and Unlearning Hyperparameters

When you run a training experiment or hyperparameter tune save the settings and results to a markdown file for the algorithm in the experiment_logs directory. Avoid creating new tables - few tables makes comparison easy. Add the baseline model evaluation results as the first row. Save rows for the settings you are about to test first then add results as soon as they're available.

Standard results columns: number of training steps, batch size, final training losses (each available separately logged loss term), MMLU accuracy, WMDP Bio Robust accuracy.

Don't vary the number of training steps on your own initiative.

When you hyperparameter tune an unlearning algorithm your first task is to find the boundary zone between where accuracy drops on both MMLU and WMDP Bio Robust, and where it drops on neither. You second task is to find a good point within that boundary zone - either where both evaluation accuracies drop partway, or where WMDP Bio Robust reduces to random while MMLU is preserved.

Unlearning hyperparameters don't transfer between number of training steps. Only comment on this if you find an exception to the rule.

Don't write "Key Findings", "Conclusions", or otherwise add your analysis to the markdown. Only record the eval results.

## Training mode

Default to SFT (full parameter training) unless LoRA is specifically requested.

## Learning rates

When training a LoRA the most common successful value is lr=1e-3. When doing SFT it's around 2e-4. Don't push SFT higher than 5e-4 without permission - if you're failing to get learning with an lr above this you likely have a bug.

# Project Structure and Conventions

Never save logs, scripts, and other development files into the root of a project. Use an appropriate directory such as `runs/` (for files with only transient value) or `unlearn/scripts/` for files to be committed.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Consider writing a new file if you add a standalone, complex feature used in more than one place.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

`torch.cuda.empty_cache()` doesn't do what you hope it will do - don't use it.

Put imports at the top of the file unless you have a very strong need to do otherwise.

Don't use try/except blocks. Use assert statements if absolutely necessary.

Don't write regular words in ALL CAPS. Don't use exclamation marks.

# Development

Use `pre-commit run --all-files` if you forget to install precommit and it doesn't run in the hook.

Open a dedicated tmux pane named "claude-commands" to run your commands so the user can monitor them.

Don't add default run path values to low-level code - if a module calls another module, the higher level module should inject a unique run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2`). The low-level code should make filenames or subdirectories within the given run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2/tamper_results`).

Don't save datasets to repository directories not in the .gitignore.

When you follow project conventions don't leave a comment saying (following project conventions) or similar drivel. More broadly, don't centre yourself or your decisions in the codebase. Only leave comments that are useful to other users. Boilerplate code should be self-documenting.

## Tests and Evaluations

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

To run the custom WMDP bio subset evals the path must be included. Here's a programmatic example, although this can also be done from the command line:

```
# Setup Tasks
include_path = "/path/to/unlearn/unlearn/lm_eval_tasks"
tm = TaskManager(verbosity="INFO", include_path=include_path)
results = simple_evaluate(
    model=lm,
    tasks=["wmdp_bio_robust"],
    task_manager=tm,
)
```

When you run the LM Eval Harness use all available GPUs on the node. You may need to launch a script using `subprocess` and capture its output.

## Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate`.

## Slurm cluster

When installing on a slurm cluster do it on a node with `srun pip install -e .` to prevent CPU-only versions of packages from being installed.

## Tamper Attacks

Two modes for tamper attacks with `run_tamper_attack_with_plot.py`:

**Short tamper (for normal unlearning runs):**
- Use `--epochs=1 --eval_every=10` (~100 steps, eval every 10)
- Purpose: Quickly demonstrate that standard unlearning is not tamper resistant
- These runs recover to baseline quickly, so long runs waste compute

**Long tamper (for tamper-resistant techniques):**
- Use `--epochs=30 --eval_every=100` (~3000 steps, eval every 100)
- Use `--eval_mmlu` to collect both WMDP and MMLU metrics
- Purpose: Compare aggressive unlearning (catastrophic forgetting) against random init or filtered model baselines
- These runs stay near random chance, so need longer runs to confirm resistance holds
- Catastrophic forgetting runs typically use `retain_coef=0` (no capability preservation)

Standard learning rate for both: `--lr=2e-5`

**Filtered model tamper attacks:**
- Use `--lr` of 2e-5, 5e-5, or 1e-4 (not higher)
- lr=2e-4 causes WMDP to drop from 34.6% to 30.5% and MMLU from 46.0% to 37.9% (constant LR, epoch 5 runs)
- lr=1e-4 with constant LR also degraded MMLU to 44.1%
- Use `--lr_scheduler_type=cosine --warmup_ratio=0.1` with `--epochs=5 --eval_every=500 --eval_mmlu`
