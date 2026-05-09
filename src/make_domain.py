import json
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Number of domains")
    parser.add_argument("--dataset", type=str, default=None, help="Target dataset name inside data/kd_datasets/")
    args = parser.parse_args()

    # Load per-class accuracy from baseline evaluation
    with open('baseline_class_metrics.json', 'r') as f:
        class_metrics = json.load(f)

    # Sort by accuracy ascending (harder classes first)
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['acc'])

    # Split into K difficulty domains (Domain 0 = hardest)
    chunks = np.array_split(sorted_classes, args.k)

    class_to_domain = {}
    print(f"{'='*50}\n[Domain Partition] {len(class_metrics)} classes -> {args.k} domains\n{'='*50}")

    for domain_id, chunk in enumerate(chunks):
        acc_list = [item[1]['acc'] for item in chunk]
        print(f"Domain {domain_id:02d} | {len(chunk):3d} classes | Mean Acc: {np.mean(acc_list)*100:.2f}%")
        for item in chunk:
            class_id = str(item[0])
            class_to_domain[class_id] = int(domain_id)

    save_name = f'class_to_domain_{args.k}.json'
    if args.dataset:
        save_path = os.path.join('data/kd_datasets', args.dataset, save_name)
    else:
        save_path = save_name

    with open(save_path, 'w') as f:
        json.dump(class_to_domain, f, indent=4)
        
    print(f"\nDomain map saved to: {save_path}")
    if not args.dataset:
        print("NOTE: Move this JSON file to your dataset root directory.")

if __name__ == "__main__":
    main()