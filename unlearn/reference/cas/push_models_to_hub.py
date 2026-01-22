import gc

from utils import *

if __name__ == "__main__":

    models = [
        # 'Unlearning/pythia1.5_baseline',
        # 'Unlearning/pythia1.5_modernbert_filtered_5percent_replace_with_escelations',
        # 'Unlearning/pythia1.5_blocklist_filtered',
        "models/Unlearning/pythia1.5_baseline_rr_new",
        "models/Unlearning/pythia1.5_modernbert_filtered_5percent_replace_with_escelations_rr_new",
        "models/Unlearning/pythia1.5_blocklist_filtered_rr_new",
        "models/Unlearning/pythia1.5_baseline_rr-lat_new",
        "models/Unlearning/pythia1.5_modernbert_filtered_5percent_replace_with_escelations_rr-lat_new",
        "models/Unlearning/pythia1.5_blocklist_filtered_rr-lat_new",
    ]

    save_names = [
        # '',
        # '',
        # '',
        "Unlearning/deep-ignorance-cb-7b",
        "Unlearning/deep-ignorance-weak-filter-cb-7b",
        "Unlearning/deep-ignorance-strong-filter-cb-7b",
        "Unlearning/deep-ignorance-cb-lat-7b",
        "Unlearning/deep-ignorance-weak-filter-cb-lat-7b",
        "Unlearning/deep-ignorance-strong-filter-cb-lat-7b",
    ]

    for model_name, save_name in zip(models, save_names):

        if (
            "rmu" in model_name
            or "5percent" in model_name
            or "lie_o" in model_name
            or "smollm" in model_name
            or "deepfried" in model_name
            or "deep_fry" in model_name
            or "replace" in model_name
        ):
            revision = "main"
        else:
            revision = "annealing_step_11921"

        model, tokenizer = get_model_and_tokenizer(model_name, revision=revision)

        # print(f"Loaded model {model_name} with revision {revision}.")
        # ask_simple_questions(model, tokenizer)
        # print('\n\n\n')

        model.push_to_hub(save_name, private=False)
        print(f"Model {model_name} pushed to Hugging Face Hub as {save_name}.")
        del model, tokenizer
        gc.collect()
