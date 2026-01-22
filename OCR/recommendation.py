import json
import re
from typing import Dict, List, Any


SCORE_THRESHOLD = 0.2


def find_model_matches(
    anon_path: str = 'test_list_original.txt',
    models_path: str = 'models.json',
    score_threshold: float = SCORE_THRESHOLD,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Match OCR text data against model definitions and return ranked matches.
    
    Args:
        anon_path: Path to the anonymous text data file collected through OCR
        models_path: Path to the JSON file containing model definitions
        score_threshold: Minimum score threshold for including matches (default: 0.5)
        verbose: If True, print results to console (default: True)
    
    Returns:
        List of model matches sorted by score (descending), each containing:
        - model_uuid: UUID of the matched model
        - score: Match score (0-1)
        - matches: Dictionary of matched identifiers and their captured values
    """
    # Read anonymous text data
    try:
        with open(anon_path, 'r', encoding='utf-8') as f:
            anon_data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Anonymous text data file not found: {anon_path}")
    
    # Load model definitions
    try:
        with open(models_path, 'r', encoding='utf-8') as f:
            models = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Models JSON file not found: {models_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in models file: {e}")
    
    model_matches = []
    
    # Each model
    for model in models:
        match_count = 0
        matches = {}
        
        # Each identifier
        identifiers = model.get('identifiers', {})
        for key, pattern in identifiers.items():
            print(f"Matching pattern {pattern} against text blob {anon_data}")
            # Match pattern against text blob
            # Ruby's =~ returns match object, $1 is first capture group
            # if pattern == "^SN:\\s(.*)$":
            #     print(f"SN: {anon_data}")
            #     res = re.search(pattern, anon_data, re.MULTILINE)
            #     print(f"applied pattern {pattern} to text blob {anon_data}, Result: {res}")
            #     break
            match = re.search(pattern, anon_data, re.MULTILINE)
            print("\n\n")
            print(f"value of match: {match}")
            print("\n\n")
            if match:
                # If matches, increment tally and add result to match collection
                match_count += 1
                print("_________________")
                print("\n")
                print(f"match_count is now {match_count}")
                print("\n")
                print("_________________")
                # $1 in Ruby is the first capture group, or the whole match if no groups
                captured_value = match.group(1) if match.lastindex else match.group(0)
                matches[key] = captured_value
        
        # Calculate overall match score 0-1
        identifier_count = len(identifiers)
        score = match_count / identifier_count if identifier_count > 0 else 0
        print("#######")
        print("\n")
        print(f"score is now {score}")
        print("\n")
        print("#######")
        # Model match summary
        if score > score_threshold:
            model_matches.append({
                'model_uuid': model.get('UUID'),
                'score': score,
                'matches': matches
            })
    
    # Sort by match score (descending)
    model_matches.sort(key=lambda m: m['score'], reverse=True)
    
    if verbose:
        print("Ranked Model Matches:")
        for i, m in enumerate(model_matches, 1):
            print(f"{i}: {m}")
        
        if len(model_matches) > 0:
            # Find the top match's description
            top_match_uuid = model_matches[0]['model_uuid']
            top_model = next((m for m in models if m.get('UUID') == top_match_uuid), None)
            if top_model:
                print(f"\nTop Match: {top_model.get('description', 'N/A')}")
            else:
                print("\nTop Match: Description not found")
        else:
            print("\nNo Matches!")
    
    return model_matches


if __name__ == '__main__':
    # Example usage
    find_model_matches()
