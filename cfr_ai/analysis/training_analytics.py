import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from typing import Dict, Any

def format_value(value: Any, precision: int = 3, is_percent: bool = False) -> str:
    """Safely formats a value to a string with a given precision."""
    try:
        num = float(value)
        if is_percent:
            return f"{num:.2%}"
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return 'N/A'

def get_strategy_folder_size(directory_path: str) -> int:
    """
    Calculates the total size of all non-diagnostic strategy sub-folders.
    """
    total_size = 0
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        # Look for directories that are purely numeric (like '1')
        if os.path.isdir(item_path) and item_name.isdigit():
            for dirpath, _, filenames in os.walk(item_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024)) # Convert to MB

def parse_metadata(file_path: str) -> Dict[str, Any]:
    """Parses a metadata.csv file into a dictionary."""
    data = {}
    utility_log = {'iter': [], 'p0': [], 'p1': []}
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            key, value = row
            # --- Utility Log Parsing ---
            match = re.search(r"P(\d) Utility at Iter (\d+)", key)
            if match:
                player = int(match.group(1))
                iteration = int(match.group(2))
                if player == 0:
                    # Add new entry if this is a new iteration log point
                    if iteration not in utility_log['iter']:
                        utility_log['iter'].append(iteration)
                        utility_log['p0'].append(float(value))
                        # Add a placeholder for p1 that will be filled later
                        utility_log['p1'].append(None)
                elif player == 1:
                    # Find the corresponding p0 entry and add the p1 value
                    idx = utility_log['iter'].index(iteration)
                    utility_log['p1'][idx] = float(value)

            # --- Regular Metadata Parsing ---
            else:
                data[key.strip()] = value.strip()
    
    data['utility_log'] = utility_log
    return data

def generate_utility_chart(setup_name: str, data: Dict[str, Any], output_dir: str) -> None:
    """Creates and saves a utility chart for a single run."""
    log = data.get('utility_log')
    if not log or not log['iter']:
        print(f"No utility data to plot for {setup_name}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(log['iter'], log['p0'], marker='o', linestyle='-', label='Player 0 Utility')
    plt.plot(log['iter'], log['p1'], marker='o', linestyle='-', label='Player 1 Utility')
    
    plt.title(f'Utility During Training for Setup {setup_name}')
    plt.xlabel('Training Iteration')
    plt.ylabel('Average Utility in Chunk')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, f'utility_chart_{setup_name}.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved utility chart to {chart_path}")


def main() -> None:
    """
    Main function to scan directories, parse data, create the summary table,
    and generate charts.
    """
    root_dir = 'cfr_ai/outputs'
    all_runs_data = []

    for subdir, _, files in os.walk(root_dir):
        if 'metadata.csv' in files:
            file_path = os.path.join(subdir, 'metadata.csv')
            setup_name = os.path.basename(subdir)
            
            data = parse_metadata(file_path)
            
            # Create Summary Table Row
            p0_value = format_value(data.get('Player 1 game value', 'N/A'))
            p1_value = format_value(data.get('Player 2 game value', 'N/A'))
            exp_value_str = f"{p0_value}, {p1_value}"

            # Exploitability is now produced by cfr_ai/lbr.py and lives in
            # outputs/lbr_summary.csv; no longer written into metadata.csv.

            date_str = data.get('Time finished', '')
            day_month = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S").strftime("%d.%m")

            storage_mb = get_strategy_folder_size(subdir)

            run_summary = {
                'Round': sum([int(n_cards) for n_cards in setup_name.split('_')]) - 1,
                'Setup': setup_name.replace('_', ','),
                'Day': day_month,
                'Iterations': data.get('Iterations', 'N/A'),
                'Min. bet': data.get('Minimum bet', '0'),
                'Pruning range': data.get('Pruning threshold') + ', ' + data.get('Minimum regret'),
                'Penalty': data.get('Penalty', 'N/A'),
                'Duration': data.get('Training duration', 'N/A'),
                'Exp. Value': exp_value_str,
                'Memory (MB)': round(float(data.get('RAM taken (MB)', 0))),
                'Storage (MB)': storage_mb,
                'Version': data.get('Version code', 'N/A')
            }
            all_runs_data.append(run_summary)

            # Generate Utility Chart
            generate_utility_chart(setup_name, data, subdir)

    # Display and save table
    if not all_runs_data:
        print("No metadata files found. Exiting.")
        return

    df = pd.DataFrame(all_runs_data)
    df = df.sort_values(by=['Round', 'Setup'])
    
    print("\n--- Training Summary ---")
    print(df.to_string(index=False))

    summary_csv_path = os.path.join(root_dir, 'summary_of_all_runs.csv')
    df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to {summary_csv_path}")


if __name__ == '__main__':
    main()
