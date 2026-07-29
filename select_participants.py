import os
import sys
import json
import pandas as pd
import random

def main():
    percentage = 100.0
    if len(sys.argv) > 1:
        try:
            percentage = float(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid percentage '{sys.argv[1]}'", file=sys.stderr)
            sys.exit(1)
            
    if percentage <= 0.0 or percentage > 100.0:
        print(f"Error: Percentage must be in (0, 100], got {percentage}", file=sys.stderr)
        sys.exit(1)
    
    # Read both official CSVs to get the union of all participants
    df_train = pd.read_csv('epic-kitchens-100-annotations/EPIC_100_train.csv')
    df_val = pd.read_csv('epic-kitchens-100-annotations/EPIC_100_validation.csv')
    
    p_train = set(df_train.participant_id.unique())
    p_val = set(df_val.participant_id.unique())
    
    all_participants = sorted(list(p_train | p_val)) # 34 participants
    
    # Shuffle deterministically
    random.Random(42).shuffle(all_participants)
    
    # Partition them into Train (80%), Val (10%), Test (10%) pools
    n_total = len(all_participants)
    n_train = int(round(n_total * 0.8))
    n_val = int(round(n_total * 0.1))
    
    train_pool = all_participants[:n_train]
    val_pool = all_participants[n_train:n_train+n_val]
    test_pool = all_participants[n_train+n_val:]
    
    # Scale each pool by the requested percentage
    n_train_sel = max(1, int(round(len(train_pool) * percentage / 100.0)))
    n_val_sel = max(1, int(round(len(val_pool) * percentage / 100.0)))
    n_test_sel = max(1, int(round(len(test_pool) * percentage / 100.0)))
    
    selected_train = train_pool[:n_train_sel]
    selected_val = val_pool[:n_val_sel]
    selected_test = test_pool[:n_test_sel]
    
    # Save selected participants to selected_participants.json
    selected_dict = {
        'train': selected_train,
        'val': selected_val,
        'test': selected_test
    }
    with open('selected_participants.json', 'w') as f:
        json.dump(selected_dict, f, indent=2)
        
    all_selected = selected_train + selected_val + selected_test
    print(','.join(all_selected))

if __name__ == '__main__':
    main()
