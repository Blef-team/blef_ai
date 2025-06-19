import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

def format_value(value, precision=3, is_percent=False):
    """Safely formats a value to a string with a given precision."""
    try:
        num = float(value)
        if is_percent:
            return f"{num:.2%}"
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return 'N/A'

def parse_metadata(file_path):
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

def generate_utility_chart(setup_name, data, output_dir):
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


def main():
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

            exploit_p0 = format_value(data.get('Exploitability when player 0 starts', 'N/A'), is_percent=True)
            exploit_p1 = format_value(data.get('Exploitability when player 1 starts', 'N/A'), is_percent=True)
            exploit_str = exploit_p0
            if exploit_p1 != 'N/A':
                exploit_str += f", {exploit_p1}"

            date_str = data.get('Time finished', '')
            day_month = datetime.strptime(date_str, "%Y-%m-%d, %H:%M:%S").strftime("%d.%m")

            run_summary = {
                'Setup': setup_name.replace('_', ','),
                'Day': day_month,
                'Iterations': data.get('Iterations', 'N/A'),
                'Duration': data.get('Training duration', 'N/A'),
                'Pruning range': data.get('Pruning threshold') + ', ' + data.get('Minimum regret'),
                'Exp. Value': exp_value_str,
                'Memory (MB)': round(float(data.get('RAM taken (MB)', 0))),
                'Penalty': data.get('Penalty', 'N/A'),
                'Exploitability': exploit_str
            }
            all_runs_data.append(run_summary)

            # Generate Utility Chart
            generate_utility_chart(setup_name, data, subdir)

    # Display and save table
    if not all_runs_data:
        print("No metadata files found. Exiting.")
        return

    df = pd.DataFrame(all_runs_data)
    df = df.sort_values(by='Setup')
    
    print("\n--- Training Summary ---")
    print(df.to_string(index=False))

    summary_csv_path = os.path.join(root_dir, 'summary_of_all_runs.csv')
    df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to {summary_csv_path}")


if __name__ == '__main__':
    main()
