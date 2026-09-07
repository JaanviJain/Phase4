import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_scifact(raw_dir, output_dir):
    corpus_file = os.path.join(raw_dir, "scifact", "corpus.jsonl")
    
    if not os.path.exists(corpus_file):
        print(f"WARNING: SciFact corpus not found at {corpus_file}. Skipping.")
        return None
    
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc['doc_id'])
            title = doc.get('title', '')
            abstract = ' '.join(doc.get('abstract', []))
            corpus[doc_id] = f"{title} {abstract}".strip()
    
    claim_files = [
        ('train', os.path.join(raw_dir, "scifact", "claims_train.jsonl")),
        ('dev', os.path.join(raw_dir, "scifact", "claims_dev.jsonl")),
        ('test', os.path.join(raw_dir, "scifact", "claims_test.jsonl"))
    ]
    
    data = []
    for split_name, claims_file in claim_files:
        if not os.path.exists(claims_file):
            print(f"WARNING: SciFact {split_name} not found at {claims_file}. Skipping.")
            continue
        
        count = 0
        with open(claims_file, 'r', encoding='utf-8') as f:
            for line in f:
                claim = json.loads(line)
                claim_text = claim['claim']
                
                evidence_texts = []
                for doc_id in claim.get('cited_doc_ids', []):
                    doc_text = corpus.get(str(doc_id), "")
                    if doc_text:
                        evidence_texts.append(doc_text)
                
                evidence_str = ' '.join(evidence_texts) if evidence_texts else "No evidence provided."
                
                label_str = claim.get('label', None)
                
                if label_str is None:
                    evidence_dict = claim.get('evidence', {})
                    if evidence_dict:
                        first_doc_id = list(evidence_dict.keys())[0]
                        first_ev = evidence_dict[first_doc_id][0]
                        label_str = first_ev.get('label', 'NOT ENOUGH INFO')
                    else:
                        label_str = 'NOT ENOUGH INFO'
                
                label_str = str(label_str).upper().strip()
                
                if label_str in ["SUPPORT", "SUPPORTS"]:
                    label = 2
                elif label_str in ["CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES"]:
                    label = 0
                else:
                    label = 1
                
                data.append({
                    'claim': claim_text,
                    'evidence': evidence_str,
                    'label': label,
                    'source': 'scifact'
                })
                count += 1
        
        print(f"SciFact {split_name}: {count} claims loaded")
    
    if not data:
        print("WARNING: No SciFact data loaded.")
        return None
    
    df = pd.DataFrame(data)
    out_path = os.path.join(output_dir, "scifact_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"SciFact TOTAL: {len(df)} samples -> {out_path}")
    print(f"  Labels: {df['label'].value_counts().sort_index().to_dict()}")
    return df


def preprocess_healthfc(raw_dir, output_dir):
    csv_path = os.path.join(raw_dir, "healthfc", "Datensatz.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found. Skipping HealthFC.")
        return None
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"HealthFC columns: {list(df.columns)}")
    
    claim_col = [c for c in df.columns if 'claim' in c.lower()][0] if any('claim' in c.lower() for c in df.columns) else df.columns[0]
    evidence_col = [c for c in df.columns if 'evidence' in c.lower()][0] if any('evidence' in c.lower() for c in df.columns) else df.columns[1]
    label_col = [c for c in df.columns if any(x in c.lower() for x in ['label', 'verdict'])][0] if any(any(x in c.lower() for x in ['label', 'verdict']) for c in df.columns) else df.columns[2]
    
    def map_label(v):
        v = str(v).lower().strip()
        if v in ['true', 'wahr', 'richtig', 'supported', 'correct', '2']:
            return 2
        elif v in ['false', 'falsch', 'refuted', 'incorrect', 'wrong', '0']:
            return 0
        else:
            return 1
    
    df['label'] = df[label_col].apply(map_label)
    df['claim'] = df[claim_col]
    df['evidence'] = df[evidence_col]
    df['source'] = 'healthfc'
    
    out_df = df[['claim', 'evidence', 'label', 'source']].copy()
    out_path = os.path.join(output_dir, "healthfc_processed.csv")
    out_df.to_csv(out_path, index=False)
    print(f"HealthFC: {len(out_df)} samples -> {out_path}")
    print(f"  Labels: {out_df['label'].value_counts().sort_index().to_dict()}")
    return out_df


def preprocess_clinifact(raw_dir, output_dir):
    splits = {
        'train': os.path.join(raw_dir, "clinifact", "clinifact_train_set_for_phase4.csv"),
        'val': os.path.join(raw_dir, "clinifact", "clinifact_validation_set_for_phase4.csv"),
        'test': os.path.join(raw_dir, "clinifact", "clinifact_test_set_for_phase4.csv")
    }
    
    all_dfs = []
    for split_name, path in splits.items():
        if not os.path.exists(path):
            print(f"WARNING: CliniFact {split_name} not found at {path}. Skipping.")
            continue
        
        df = pd.read_csv(path)
        print(f"CliniFact {split_name}: {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Labels: {df['label'].value_counts().sort_index().to_dict()}")
        
        required = {'claim', 'evidence', 'label', 'source'}
        missing = required - set(df.columns)
        if missing:
            print(f"  ERROR: Missing columns {missing}. Skipping this split.")
            continue
        
        all_dfs.append(df)
    
    if not all_dfs:
        print("WARNING: No CliniFact splits loaded.")
        return None
    
    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(output_dir, "clinifact_processed.csv")
    combined.to_csv(out_path, index=False)
    print(f"CliniFact combined: {len(combined)} samples -> {out_path}")
    return combined


def preprocess_pubhealth(raw_dir, output_dir):
    tsv_path = os.path.join(raw_dir, "pubhealth", "train.tsv")
    if not os.path.exists(tsv_path):
        print(f"WARNING: {tsv_path} not found. Skipping PUBHEALTH.")
        return None
    
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
    print(f"PUBHEALTH columns: {list(df.columns)}")
    
    claim_col = [c for c in df.columns if 'claim' in c.lower()][0] if any('claim' in c.lower() for c in df.columns) else df.columns[0]
    evidence_col = [c for c in df.columns if 'explanation' in c.lower()][0] if any('explanation' in c.lower() for c in df.columns) else None
    if evidence_col is None:
        evidence_col = [c for c in df.columns if 'evidence' in c.lower()][0] if any('evidence' in c.lower() for c in df.columns) else df.columns[1]
    label_col = [c for c in df.columns if 'label' in c.lower()][0] if any('label' in c.lower() for c in df.columns) else df.columns[-1]
    
    def map_label(v):
        v = str(v).lower().strip()
        if v == 'true':
            return 2
        elif v == 'false':
            return 0
        elif v == 'unproven':
            return 1
        elif v == 'mixture':
            return 0
        else:
            return 1
    
    df['label'] = df[label_col].apply(map_label)
    df['claim'] = df[claim_col]
    df['evidence'] = df[evidence_col].fillna("No explanation provided.")
    df['source'] = 'pubhealth'
    
    out_df = df[['claim', 'evidence', 'label', 'source']].copy()
    out_path = os.path.join(output_dir, "pubhealth_processed.csv")
    out_df.to_csv(out_path, index=False)
    print(f"PUBHEALTH: {len(out_df)} samples -> {out_path}")
    print(f"  Labels: {out_df['label'].value_counts().sort_index().to_dict()}")
    return out_df


def split_and_save(df, output_dir, name):
    if df is None or len(df) == 0:
        return
    
    for lbl in [0, 1, 2]:
        if lbl not in df['label'].values:
            print(f"WARNING: {name} is missing label {lbl}.")
    
    train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    train, val = train_test_split(train_val, test_size=0.176, random_state=42, stratify=train_val['label'])
    
    train.to_csv(os.path.join(output_dir, f"{name}_train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, f"{name}_val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, f"{name}_test.csv"), index=False)
    
    print(f"  {name} split: Train={len(train)}, Val={len(val)}, Test={len(test)}")


def combine_splits(output_dir):
    for split in ['train', 'val', 'test']:
        files = [f for f in os.listdir(output_dir) if f.endswith(f"_{split}.csv")]
        if not files:
            continue
        
        dfs = [pd.read_csv(os.path.join(output_dir, f)) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        out_path = os.path.join(output_dir, f"combined_{split}.csv")
        combined.to_csv(out_path, index=False)
        
        print(f"\nCombined {split}: {len(combined)} total samples -> {out_path}")
        print(f"  By source: {combined['source'].value_counts().to_dict()}")
        print(f"  By label:  {combined['label'].value_counts().sort_index().to_dict()}")


def main():
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    print("="*60)
    print("PHASE 4 DATA PREPROCESSING")
    print("="*60)
    
    scifact_df = preprocess_scifact(raw_dir, processed_dir)
    healthfc_df = preprocess_healthfc(raw_dir, processed_dir)
    clinifact_df = preprocess_clinifact(raw_dir, processed_dir)
    pubhealth_df = preprocess_pubhealth(raw_dir, processed_dir)
    
    print("\nSplitting datasets...")
    split_and_save(scifact_df, processed_dir, 'scifact')
    split_and_save(healthfc_df, processed_dir, 'healthfc')
    split_and_save(clinifact_df, processed_dir, 'clinifact')
    split_and_save(pubhealth_df, processed_dir, 'pubhealth')
    
    print("\nCreating combined splits...")
    combine_splits(processed_dir)
    
    print("\nPreprocessing complete! You can now run: python train.py")


if __name__ == "__main__":
    main()