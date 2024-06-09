from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from vllm import LLM, SamplingParams

from . import templates


def load_mmlu(data_dir="data"):
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


def get_prompts(records):
    sys_template = templates.zero_shot_system_prompt
    user_template = templates.zero_shot_mmlu_prompt
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


ABCD_PAT = re.compile(r"\b([A-D])\b")


def get_pred(reponse):
    match = ABCD_PAT.search(completion)
    if match:
        pred = match.group(1)
    else:
        pred = None
    return pred


def get_preds(responses):
    preds = []
    for response in responses:
        pred = get_pred(response)
        preds.append(pred)
    return preds


def get_accuracy(answers, preds):
    total = len(answers)
    correct = 0
    for i in range(total):
        if answers[i] == preds[i]:
            correct += 1
    accuracy = correct / total
    return accuracy


def get_mmlu_score(records, preds):
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
