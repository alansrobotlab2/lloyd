#!/usr/bin/env python3
"""
Fact-content similarity sweep: Find entities with highly similar fact corpora.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR

def normalize_fact(fact: str) -> str:
    """Normalize fact text for comparison."""
    fact = fact.lower()
    fact = re.sub(r'[^\w\s]', '', fact)  # Remove punctuation
    fact = re.sub(r'\s+', ' ', fact).strip()  # Collapse whitespace
    return fact

def load_facts():
    """Load all facts from the facts directory."""
    entity_facts = defaultdict(list)
    
    for entity_dir in FACTS_DIR.iterdir():
        if not entity_dir.is_dir() or entity_dir.name.startswith('.'):
            continue
        
        facts_file = entity_dir / "facts.jsonl"
        if not facts_file.exists():
            continue
        
        with open(facts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact_data = json.loads(line)
                    fact_text = fact_data.get('fact', '')
                    if fact_text:
                        entity_facts[entity_dir.name].append(normalize_fact(fact_text))
                except json.JSONDecodeError:
                    continue
    
    return entity_facts

def compute_similarity_matrix(entity_facts):
    """Compute pairwise fact corpus similarity using TF-IDF."""
    entities = list(entity_facts.keys())
    n = len(entities)
    
    # Combine facts for each entity into a single document
    documents = [' '.join(facts) for facts in entity_facts.values()]
    
    # Filter out entities with too few facts
    valid_indices = [i for i, facts in enumerate(entity_facts.values()) if len(facts) >= 3]
    valid_entities = [entities[i] for i in valid_indices]
    valid_docs = [documents[i] for i in valid_indices]
    
    if len(valid_docs) < 2:
        return [], []
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(valid_docs)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Extract pairs with high similarity
    candidates = []
    for i in range(len(valid_entities)):
        for j in range(i + 1, len(valid_entities)):
            sim = similarity_matrix[i, j]
            if sim >= 0.75:
                candidates.append({
                    'entity_a': valid_entities[i],
                    'entity_b': valid_entities[j],
                    'similarity': float(sim),
                    'facts_a': len(entity_facts[valid_entities[i]]),
                    'facts_b': len(entity_facts[valid_entities[j]])
                })
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    return candidates

def main():
    print("Loading facts...")
    entity_facts = load_facts()
    print(f"Loaded facts for {len(entity_facts)} entities")
    
    print("Computing fact-content similarity...")
    candidates = compute_similarity_matrix(entity_facts)
    
    print(f"\nFound {len(candidates)} high-similarity pairs (≥0.75)\n")
    
    # Show top 20
    for i, c in enumerate(candidates[:20]):
        print(f"{i+1}. Similarity={c['similarity']:.3f} | {c['entity_a']} ↔ {c['entity_b']}")
        print(f"   Facts: {c['facts_a']}/{c['facts_b']}")
        print()
    
    # Save results
    output_file = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "fact-similarity-candidates.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
