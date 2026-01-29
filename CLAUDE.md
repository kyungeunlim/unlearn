Always test your changes by running the appropriate script or CLI command. Never complete a task without testing your changes using a script or CLI command. To test, run the script and then frequently monitor the output until it appears to be running without issue, and then check again every 30 seconds until either 3 minutes have passed or multiple iteration loops of the main computation have completed successfully. If you find an error unrelated to your task, at minimum quote the exact error back to me when you have completed your task and offer to investigate and fix it.

When you make a script put it in unlearn/scripts.

## Project Structure and Conventions

Consider writing a new file if you add a standalone, complex feature used in more than one place.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

Never save logs, scripts, and other random development into the root of a project. Create an appropriate directory such as runs/ or scripts/ and add it to the .gitignore if it has only transient value.

torch.cuda.empty_cache() doesn't do what you hope it will do - don't use it.

Put imports at the top of the file unless you have a very strong need to do otherwise.

# Experiment Logs and Unlearning Hyperparameters

When you run a training experiment or hyperparameter tune save the settings and results to a markdown file for the algorithm in the experiment_logs directory. Avoid creating new tables - few tables makes comparison easy. Add the baseline model evaluation results as the first row. Save rows for the settings you are about to test first then add results when available.

Standard results columns: number of training steps, batch size, final training losses (each available separately logged loss term), MMLU accuracy, WMDP Bio Robust accuracy.

Don't vary the number of training steps on your own initiative.

When you hyperparameter tune an unlearning algorithm your first task is to find the boundary zone between where accuracy drops on both MMLU and WMDP Bio Robust, and where it drops on neither. You second task is to find a good point within that boundary zone - either where both evaluation accuracies drop partway, or where WMDP Bio Robust reduces to random while MMLU is preserved.

# Development

Use `pre-commit run --all-files` if you forget to install pre-commit and it doesn't run in the hook.

Open a dedicated tmux pane named "claude-commands" to run your commands so the user can monitor them.

Don't add default run path values to low-level code - if a module calls another module, the higher level module should inject a unique run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2`). The low-level code should make filenames or subdirectories within the given run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2/tamper_results`).

Don't save datasets to repository directories not in the .gitignore.

When you follow project conventions don't leave a comment saying (following project conventions) or similar drivel. More broadly, don't centre yourself or your decisions in the codebase. Only leave comments that are useful to other users. Boilerplate code should be self-documenting.

### Tests, Evals, and Hyperparameters

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

### Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate`.

## Slurm cluster

When installing on a slurm cluster do it on a node with `srun pip install -e .` to prevent CPU-only versions of packages from being installed.
