#!/usr/bin/env python3
"""Pre-download and cache all HuggingFace datasets used by lm_eval tasks.

Run this once before launching batch jobs so they can use HF_HUB_OFFLINE=1.
"""

from datasets import load_dataset

MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

WMDP_BIO_SUBSETS = [
    "bioweapons_and_bioterrorism",
    "dual_use_virology",
    "enhanced_potential_pandemic_pathogens",
    "expanding_access_to_threat_vectors",
    "reverse_genetics_and_easy_editing",
    "viral_vector_research",
]


def main():
    print(f"Caching {len(MMLU_SUBSETS)} MMLU subsets...")
    for subset in MMLU_SUBSETS:
        load_dataset("cais/mmlu", subset)
    print("MMLU done.")

    print(f"Caching {len(WMDP_BIO_SUBSETS)} WMDP Bio Robust subsets...")
    for subset in WMDP_BIO_SUBSETS:
        load_dataset("EleutherAI/wmdp_bio_robust_mcqa", subset)
    print("WMDP Bio Robust done.")

    print("All eval datasets cached.")


if __name__ == "__main__":
    main()
