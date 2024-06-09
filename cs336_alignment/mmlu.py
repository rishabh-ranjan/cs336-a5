import csv
from pathlib import Path
import pkg_resources
import re

from vllm import LLM, SamplingParams


def parse_mmlu(data_dir="data"):
    test_dir = f"{data_dir}/mmlu/test"
    subject_csv_files = list(Path(test_dir).iterdir())
    records = []
    for subject_csv_file in subject_csv_files:
        subject = subject_csv_file.stem[: -len("_test")]
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


def mmlu_prompts(records):
    prompt_file = pkg_resources.resource_filename(
        __name__, "prompts/zero_shot_mmlu_prompt.prompt"
    )
    with open(prompt_file, "r") as f:
        template = f.read()
    prompts = []
    for record in records:
        prompt = template.format(**record)
        prompts.append(prompt)
    return prompts


def infer(prompts):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,  # XXX
        stop=["\n"],  # XXX
    )

    llm = LLM(model="meta-llama/Meta-Llama-3-8B")

    outputs = llm.generate(prompts, sampling_params)

    completions = []
    for output in outputs:
        prompt = output.prompt
        completion = output.outputs[0].text
        completions.append(completion)

    return completions


abcd_pat = re.compile(r"\b([A-D])\b")


def parse_mmlu_completion(completion):
    match = abcd_pat.search(completion)
    if match:
        answer = match.group(1)
    else:
        answer = None
    return answer
