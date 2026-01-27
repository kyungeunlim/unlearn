import os
import subprocess

import torch
import wandb
from transformers import TrainerCallback

from unlearn.utils.math import effective_rank, stable_rank


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        run_every_steps=50,
        ref_model=None,
        tasks=["wmdp_bio_robust", "mmlu_stem", "wmdp_bio_cloze_verified"],
        include_path=None,
    ):
        self.tokenizer = tokenizer
        self.run_every_steps = run_every_steps

        self.ref_state_dict = {}
        if ref_model:
            for name, param in ref_model.named_parameters():
                if param.dim() > 1:  # Only track matrices (weights), not biases
                    self.ref_state_dict[name] = (
                        param.detach().cpu().clone().to(torch.float32)
                    )
        self.include_path = include_path
        self.tasks = tasks

        # Calculate the stable rank of the initial parameters
        ranks = self._compute_module_ranks()
        mean = torch.tensor(ranks, dtype=torch.float32).mean().item()
        std = torch.tensor(ranks, dtype=torch.float32).std().item()
        print(
            f"Initial Mean Stable Rank of Modules: {mean:.4f}, Std: {std:.4f}"
            f"Max: {max(ranks):.4f}, Min: {min(ranks):.4f}"
        )

    def _run_evaluation(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            model = kwargs["model"]
            checkpoint_path = f"{args.output_dir}/eval_checkpoint_{state.global_step}"
            model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)

            env = os.environ.copy()
            env.update(
                {
                    "STEP": str(state.global_step),
                    "OUTPUT_DIR": args.output_dir,
                    "CHECKPOINT_PATH": checkpoint_path,
                    "RESULTS_PATH": (
                        f"{args.output_dir}/eval_results_{state.global_step}"
                    ),
                    "TASKS": ",".join(self.tasks),
                    "INCLUDE_PATH": str(self.include_path),
                    "WANDB_RUN_ID": wandb.run.id,  # type: ignore
                    "WANDB_PROJECT": wandb.run.project,  # type: ignore
                }
            )

            result = subprocess.run(
                ["sbatch", "unlearn/scripts/eval_job.sh"],
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode == 0:
                print(
                    f"[Step {state.global_step}] "
                    f"Submitted eval job: {result.stdout.strip()}",
                    flush=True,
                )
            else:
                print(
                    f"[Step {state.global_step}] "
                    f"Failed to submit eval job: {result.stderr}",
                    flush=True,
                )

            print("Calculating Stable Rank of parameter update...")
            if self.ref_state_dict:
                param_stats = self._compute_param_stats(model)
                stable_ranks = param_stats["stable_ranks"]
                effective_ranks = param_stats["effective_ranks"]
                frob_norm = param_stats["frob_norm"]

                mean_stable_rank = (
                    torch.tensor(stable_ranks, dtype=torch.float32).mean().item()
                )
                mean_effective_rank = (
                    torch.tensor(effective_ranks, dtype=torch.float32).mean().item()
                )

                wandb.log(
                    {
                        "update_stable_rank": mean_stable_rank,
                        "update_effective_rank": mean_effective_rank,
                        "update_stable_rank_dist": wandb.Histogram(stable_ranks),
                        "update_effective_rank_dist": wandb.Histogram(effective_ranks),
                        "update_frob_norm ": frob_norm,
                    },
                    step=state.global_step,
                )

    def _compute_module_ranks(self):
        """Computes the stable rank of each module."""
        ranks = []
        with torch.no_grad():
            for param in self.ref_state_dict.values():
                # Only track matrices (weights), not biases
                if param.dim() <= 1:
                    continue

                ranks.append(stable_rank(param))
        return ranks

    def _compute_param_stats(self, current_model):
        """Computes the average stable rank of (W_now - W_init)
        and the frobenius norm of the difference."""
        mod_stable_ranks = []
        mod_effective_ranks = []

        # For frobenius norm
        ssqs = torch.tensor(0.0, dtype=torch.float32)

        with torch.no_grad():
            for name, param in current_model.named_parameters():
                if name in self.ref_state_dict:
                    ref_param = self.ref_state_dict[name].to(device=param.device)
                    if param.dim() <= 1:
                        continue

                    diff = param.to(torch.float32) - ref_param

                    mod_stable_ranks.append(stable_rank(diff))
                    mod_effective_ranks.append(effective_rank(diff))

                    # Accumulate sum of squares
                    ssq = diff.pow(2).sum()
                    ssqs += ssq.to(ssqs.device)

        frob_norm = ssqs.sqrt()

        return {
            "stable_ranks": mod_stable_ranks,
            "effective_ranks": mod_effective_ranks,
            "frob_norm": frob_norm,
        }

    def on_train_begin(self, args, state, control, **kwargs):
        self._run_evaluation(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.run_every_steps == 0 and state.global_step > 0:
            self._run_evaluation(args, state, control, **kwargs)
