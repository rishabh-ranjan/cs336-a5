from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from vllm import LLM, SamplingParams

from . import templates


def get_records(eval_, data_dir="data"):
    if eval_ == "mmlu":
        test_dir = f"{data_dir}/mmlu/test"
        subject_csv_files = list(Path(test_dir).iterdir())
        records = []
        for subject_csv_file in subject_csv_files:
            subject = Path(subject_csv_file).stem[: -len("_test")]
            subject = subject.replace("_", " ")
            with open(subject_csv_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    record = {
                        "subject": subject,
                        "question": row[0],
                        "options": row[1:5],
                        "answer": row[5],
                    }
                    records.append(record)
        return records

    if eval_ == "gsm8k":
        test_file = f"{data_dir}/gsm8k/test.jsonl"
        with open(test_file, "r") as f:
            records = []
            for line in f:
                record = json.loads(line)
                records.append(record)
        return records

    if eval_ == "alpaca_eval":
        file = f"{data_dir}/alpaca_eval/alpaca_eval.jsonl"
        with open(file, "r") as f:
            records = []
            for line in f:
                record = json.loads(line)
                records.append(record)
        return records


def get_prompts(eval_, records):
    sys_template = templates.system_prompt
    user_template = templates.user_prompt_for_eval[eval_]
    prompts = []
    for record in records:
        instruction = user_template.format(**record)
        prompt = sys_template.format(instruction=instruction)
        prompts.append(prompt)
    return prompts


def get_responses(prompts):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["# Query:"],
    )
    llm = LLM(model="meta-llama/Meta-Llama-3-8B")
    outputs = llm.generate(prompts, sampling_params)

    responses = []
    for output in outputs:
        response = output.outputs[0].text
        responses.append(response)

    return responses


def dump_alpaca_eval_responses(records, responses, out_file):
    out_records = []
    for record, response in zip(records, responses):
        out_record = {
            "instruction": record["instruction"],
            "output": response,
            "generator": "meta-llama/Meta-Llama-3-8B",
            "dataset": record["dataset"],
        }
        out_records.append(out_record)
    with open(out_file, "w") as f:
        json.dump(out_records, f, indent=2)


MMLU_PAT = re.compile(r"\b([A-D])\b")
GSM8K_PAT = re.compile(r"(\d+)(?!.*\d)")


def get_pred(eval_, record, response):
    if eval_ == "mmlu":
        match = MMLU_PAT.search(response)
        if match:
            return match.group()
        for i, pred in enumerate("ABCD"):
            pat = (
                r"\b" + re.escape(record["options"][i].lower().replace(".", "")) + r"\b"
            )
            if re.search(pat, response.lower()):
                return pred
        return None

    if eval_ == "gsm8k":
        if response.rstrip().endswith("?```"):
            return None
        response = response.replace("\n", " ")
        response = response.replace(",", "")
        match = GSM8K_PAT.search(response)
        if match:
            return match.group()
        return None


def get_preds(eval_, records, responses):
    preds = []
    for record, response in zip(records, responses):
        pred = get_pred(eval_, record, response)
        preds.append(pred)
    return preds


def get_score(eval_, records, preds):
    if eval_ == "mmlu":
        subject_total = defaultdict(int)
        subject_correct = defaultdict(int)
        for record, pred in zip(records, preds):
            subject = record["subject"]
            subject_total[subject] += 1
            if record["answer"] == pred:
                subject_correct[subject] += 1
        subject_acc = {}
        for subject in subject_total:
            subject_acc[subject] = subject_correct[subject] / subject_total[subject]
        avg_acc = sum(subject_acc.values()) / len(subject_acc)
        return avg_acc

    elif eval_ == "gsm8k":
        correct = 0
        for record, pred in zip(records, preds):
            gold = get_pred("gsm8k", record, record["answer"])
            if gold == pred:
                correct += 1
        acc = correct / len(records)
        return acc
