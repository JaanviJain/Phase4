import json

with open("data/raw/scifact/claims_train.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break

        claim = json.loads(line)

        print(f"Label: {claim.get('label', 'NO LABEL')}")

