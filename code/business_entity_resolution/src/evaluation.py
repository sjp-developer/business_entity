"""
Official Metric Evaluator for Amazon ML Challenge 2026.
Calculates per-S1 Macro F0.5 Score, Precision, Recall, Candidate Recall, and Singleton metrics.
"""
from typing import Dict, List, Set, Tuple

def evaluate_s1_entity(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Evaluates metric for a single Source 1 entity.
    Returns: (f05_score, precision, recall)
    """
    is_singleton = len(true_matches) == 0
    pred_count = len(pred_matches)
    
    if is_singleton:
        if pred_count == 0:
            return 1.0, 1.0, 1.0 # Correct singleton prediction
        else:
            return 0.0, 0.0, 0.0 # False merge on singleton
            
    if pred_count == 0:
        return 0.0, 0.0, 0.0 # Missed match
        
    intersection = len(true_matches.intersection(pred_matches))
    precision = intersection / pred_count
    recall = intersection / len(true_matches)
    
    denom = (0.25 * precision) + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return f05, precision, recall

def evaluate_predictions(gt_map: Dict[str, Set[str]], pred_map: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Computes macro-averaged metrics across all Source 1 entities.
    
    Args:
      gt_map: {s1_entity_id: set_of_true_matched_ids}
      pred_map: {s1_entity_id: set_of_predicted_matched_ids}
      
    Returns dictionary with:
      macro_f05, macro_precision, macro_recall, singleton_acc, total_s1
    """
    total_entities = len(gt_map)
    if total_entities == 0:
        return {}
        
    f05_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    
    singleton_count = 0
    correct_singletons = 0
    
    false_positives = 0
    false_negatives = 0
    
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        
        f05, prec, rec = evaluate_s1_entity(true_set, pred_set)
        f05_sum += f05
        prec_sum += prec
        rec_sum += rec
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                correct_singletons += 1
                
        # Count false positives and false negatives
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        false_positives += fp
        false_negatives += fn
        
    macro_f05 = f05_sum / total_entities
    macro_prec = prec_sum / total_entities
    macro_rec = rec_sum / total_entities
    singleton_acc = (correct_singletons / singleton_count * 100.0) if singleton_count > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_accuracy": singleton_acc,
        "total_s1_entities": total_entities,
        "total_singletons": singleton_count,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
