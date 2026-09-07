"""
create_demo_data.py
Creates a demo CSV for multimodal inference from your Phase 1 FakeHealth data.
Run this ONCE to generate data/pipeline/fakehealth_for_demo.csv
"""

import os
import json
import pandas as pd
from pathlib import Path


def create_fakehealth_demo_csv():
    """
    You MUST customize this function based on your actual FakeHealth folder structure.
    
    Expected FakeHealth structure:
    phase1/data/fakehealth/
        ├── images/
        │   ├── img_001.jpg
        │   └── ...
        └── metadata.json   (or reviews.json, or similar)
    
    The output CSV needs columns: claim, evidence, label, image_path
    """
    
    # ========== CUSTOMIZE THESE PATHS ==========
    fakehealth_dir = "../phase1/data/fakehealth"  # Adjust to your actual Phase 1 path
    output_csv = "data/pipeline/fakehealth_for_demo.csv"
    # ===========================================
    
    os.makedirs("data/pipeline", exist_ok=True)
    
    rows = []
    
    # Try to load FakeHealth metadata
    # FakeHealth typically has: content/ folder with articles and reviews/ with labels
    
    content_dir = os.path.join(fakehealth_dir, "content")
    reviews_file = os.path.join(fakehealth_dir, "reviews", "fakehealth_reviews.json")
    
    # If you have a different structure, adjust below
    if os.path.exists(reviews_file):
        with open(reviews_file, 'r') as f:
            reviews = json.load(f)
        
        for item in reviews:
            claim = item.get('title', item.get('claim', ''))
            evidence = item.get('text', item.get('body', item.get('content', '')))
            label = 0 if str(item.get('label', item.get('rating', ''))).lower() in ['false', 'fake', '0'] else 2
            image_path = item.get('image', item.get('image_path', ''))
            
            if claim and image_path:
                rows.append({
                    'claim': claim,
                    'evidence': evidence[:2000] if evidence else "No evidence available.",
                    'label': label,
                    'image_path': os.path.basename(image_path)
                })
    
    # Alternative: If FakeHealth is organized as article folders
    elif os.path.exists(content_dir):
        for article_folder in os.listdir(content_dir)[:100]:  # Limit to 100
            article_path = os.path.join(content_dir, article_folder)
            if not os.path.isdir(article_path):
                continue
            
            # Try to find text and image
            text_file = os.path.join(article_path, "text.txt")
            img_files = list(Path(article_path).glob("*.jpg")) + list(Path(article_path).glob("*.png"))
            
            if not img_files:
                continue
            
            text = ""
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # For demo purposes, use first 100 chars as claim, rest as evidence
            lines = text.split('\n')
            claim = lines[0] if lines else "Health claim"
            evidence = '\n'.join(lines[1:]) if len(lines) > 1 else text
            
            # Determine label from folder name or content
            label = 1  # Default NEI
            if 'fake' in article_folder.lower():
                label = 0
            elif 'real' in article_folder.lower():
                label = 2
            
            rows.append({
                'claim': claim[:500],
                'evidence': evidence[:2000],
                'label': label,
                'image_path': img_files[0].name
            })
    
    if not rows:
        print("❌ Could not find FakeHealth data at the expected location.")
        print(f"   Tried: {fakehealth_dir}")
        print("\nYou need to manually create: data/pipeline/fakehealth_for_demo.csv")
        print("With columns: claim, evidence, label, image_path")
        print("\nExample content:")
        print('claim,evidence,label,image_path')
        print('"Vaccines cause autism","Large studies show no link",0,"vaccine.jpg"')
        print('"Exercise reduces heart disease","Meta-analysis confirms benefit",2,"exercise.jpg"')
        return
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Created demo CSV: {output_csv}")
    print(f"   Samples: {len(df)}")
    print(f"   Labels: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"\nNext step: Copy your FakeHealth images to: data/raw/fakehealth/images/")
    print(f"Then run: python multimodal_demo.py")


if __name__ == "__main__":
    create_fakehealth_demo_csv()