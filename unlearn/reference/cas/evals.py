import argparse

from utils import *


def eval_lab_bench(model, tokenizer):

    # model, tokenizer = get_model_and_tokenizer(
    #     'Unlearning/pythia1.5_baseline', revision='annealing_step_11921'
    # )

    litqa2 = load_dataset("futurehouse/lab-bench", "LitQA2")
    protocolqa = load_dataset("futurehouse/lab-bench", "ProtocolQA")

    def format_question(question, ideal, distractors, passage, answer):
        distractors.insert("ACBD".index(answer), ideal)
        if len(distractors) != 4:
            return None, None
        question = (
            f"PASSAGE:\n{passage}\n\nQUESTION:\n"
            f"A: {distractors[0]}\nB: {distractors[1]}\n"
            f"C: {distractors[2]}\nD: {distractors[3]}\nAnswer:"
        )
        return question, answer

    litqa_questions, litqa_answers, protocolqa_questions, protocolqa_answers = (
        [],
        [],
        [],
        [],
    )
    for i in range(len(litqa2["train"])):
        q, a = format_question(
            litqa2["train"]["question"][i],
            litqa2["train"]["ideal"][i],
            litqa2["train"]["distractors"][i],
            litqa2["train"]["key-passage"][i],
            "ABCD"[i % 4],
        )
        if q is not None:
            litqa_questions.append(q)
            litqa_answers.append(a)

    for i in range(len(protocolqa["train"])):
        q, a = format_question(
            protocolqa["train"]["question"][i],
            protocolqa["train"]["ideal"][i],
            protocolqa["train"]["distractors"][i],
            protocolqa["train"]["protocol"][i],
            "ABCD"[i % 4],
        )
        if q is not None:
            protocolqa_questions.append(q)
            protocolqa_answers.append(a)

    all_questions = litqa_questions + protocolqa_questions
    all_answers = litqa_answers + protocolqa_answers
    answer_tokens = [t[0] for t in tokenizer([" A", " B", " C", " D"]).input_ids]
    tokenizer.padding_side = "left"
    all_logits = []

    with torch.no_grad():
        for i in range(len(all_questions)):
            tokenized_question = tokenizer(
                all_questions[i], return_tensors="pt", padding="do_not_pad"
            ).to(model.device)
            outputs = model(**tokenized_question)
            logits = outputs.logits
            all_logits.append(torch.squeeze(logits[0, -1, answer_tokens]))

    correct, all = 0, 0
    for l, a in zip(all_logits, all_answers):
        if l.argmax() == "ABCD".index(a):
            correct += 1
        all += 1

    return {"litqa2&ls": correct / all}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=str, nargs="+", default="wmdp_bio_aisi")
    parser.add_argument(
        "--model_name", type=str, default="allenai/OLMo-2-1124-7B-Instruct"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--system_instruction", type=str, default=None)

    args = parser.parse_args()
    if isinstance(args.evals, str):
        args.evals = [args.evals]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    # model, tokenizer = get_model_and_tokenizer(
    #     'Unlearning/pythia1.5_baseline', revision='annealing_step_11921'
    # )

    for eval in args.evals:
        if eval == "lab_bench":
            performance = eval_lab_bench(model, tokenizer)
            print(f"***\n{eval} performance: {performance}\n***")
        elif eval == "jailbreak":
            jailbreak_score = jailbreak_eval_model(model, tokenizer, num_examples=1000)
            print(f"***\nFinal jailbreakability: {jailbreak_score}\n***")
        else:
            performance = lm_eval_model(
                model,
                task=eval,
                tokenizer=tokenizer,
                revision=args.revision,
                system_instruction=args.system_instruction,
            )
            # performance = lm_eval_model(
            #     model, task=eval, tokenizer=tokenizer, revision='annealing_step_11921'
            # )
            print(f"***\n{eval} performance: {performance}\n***")

# from utils import *
# model_name = 'Unlearning/pythia1.5_baseline'
# revision = 'annealing_step_11921'
# model, tokenizer = get_model_and_tokenizer(model_name, revision=revision)
# performance = lm_eval_model(model, task='agieval', tokenizer=tokenizer, revision=revision, limit=1)
