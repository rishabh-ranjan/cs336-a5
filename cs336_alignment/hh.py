import gzip
import pkg_resources
import json


def pretty(records):
    out = ""
    for record in records:
        for key, val in record.items():
            out += f"{key.upper()}\n"
            out += f"----------------\n"
            out += f"{val}\n\n"
        out += "================\n\n"
    return out


def get_prompt_response(text):
    t = text.split("\n\nHuman: ")[1]
    prompt, response = t.split("\n\nAssistant: ")
    return prompt, response


def main():
    data_dir = pkg_resources.resource_filename("cs336_alignment", "data")
    in_dir = f"{data_dir}/hh-rlhf"
    out_file = f"{data_dir}/hh_train.json"

    out_records = []
    for source in [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ]:
        train_file = f"{in_dir}/{source}/train.jsonl.gz"
        with gzip.open(train_file, "r") as in_f:
            for in_line in in_f:
                in_record = json.loads(in_line)

                # ignore multi-turn conversations
                # bug in dataset gives error when counting "\n\nHuman: "
                if in_record["chosen"].count("\n\nAssistant: ") > 1:
                    continue
                if in_record["rejected"].count("\n\nAssistant: ") > 1:
                    continue

                prompt, chosen = get_prompt_response(in_record["chosen"])
                _, rejected = get_prompt_response(in_record["rejected"])
                out_record = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": source,
                }
                out_records.append(out_record)

    with open(out_file, "w") as out_f:
        json.dump(out_records, out_f, indent=2)
