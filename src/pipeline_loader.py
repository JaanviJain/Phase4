import json
import pandas as pd


def load_phase3_for_ablation(json_path, condition='weighted', top_k=3):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for claim_id, claim_data in data.items():
        if isinstance(claim_data, dict) and 'evidence' in claim_data:
            ev_list = claim_data['evidence']
            claim_text = claim_data.get('claim_text', claim_id)
            true_label = claim_data.get('true_label', 1)
        elif isinstance(claim_data, list):
            ev_list = claim_data
            claim_text = claim_id
            true_label = 1
        else:
            continue
        
        if not ev_list:
            continue
        
        if condition == 'weighted':
            ev_list = sorted(ev_list, key=lambda x: x.get('weighted_score', x.get('score', 0)), reverse=True)
        else:
            ev_list = sorted(ev_list, key=lambda x: x.get('score', 0), reverse=True)
        
        top_ev = ev_list[:top_k]
        
        if condition == 'weighted':
            ev_texts = []
            for ev in top_ev:
                tier = ev.get('predicted_tier', 'UNWEIGHTED')
                txt = ev.get('text', ev.get('abstract', ''))
                ev_texts.append(f"[TIER_{tier}] {txt[:400]}")
            evidence_str = " [EVIDENCE] ".join(ev_texts)
        else:
            ev_texts = [ev.get('text', ev.get('abstract', ''))[:400] for ev in top_ev]
            evidence_str = " [EVIDENCE] ".join(ev_texts)
        
        rows.append({
            'claim': claim_text,
            'evidence': evidence_str,
            'label': int(true_label),
            'source': 'pipeline_ablation',
            'claim_id': str(claim_id),
            'condition': condition
        })
    
    df = pd.DataFrame(rows)
    print(f"[PIPELINE] Loaded {len(df)} claims from {json_path} ({condition} mode, top-{top_k})")
    return df