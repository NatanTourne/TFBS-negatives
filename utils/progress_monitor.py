import pandas as pd
import os
import time
from tqdm import tqdm
import datetime
import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Monitor model training progress.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder.")
    args = parser.parse_args()

    output_folder = args.output_folder
    file_path = os.path.join(output_folder, "model_combinations.csv")
    model_combinations_df = pd.read_csv(file_path)

    correct_neg_mode_names = {
        "dinucl_shuffled": "dinucl-shuffled",
        "dinucl_sampled": "dinucl-sampled"
    }
    correct_neg_mode_names_reverse = {v: k for k, v in correct_neg_mode_names.items()}

    def get_current_df():
        ckpt_files = [f for f in os.listdir(output_folder) if f.endswith('.ckpt')]
        updated_ckpt_files = [
            f.replace(old, new) if old in f else f
            for f in ckpt_files
            for old, new in correct_neg_mode_names.items()
            if old in f or all(k not in f for k in correct_neg_mode_names)
        ]
        # Remove duplicates if a file matches multiple keys
        updated_ckpt_files = list(dict.fromkeys(updated_ckpt_files))
        data = []
        for file in updated_ckpt_files:
            celltype = file.split("_")[0]
            TF = "_".join(file.split("_")[1:-8])
            neg_mode = file.split("_")[-8]
            CV = file.split("_")[-7]
            date = file.split("_")[-6]
            time = file.split("_")[-5]
            data.append({"Cell Type": celltype, "TF": TF, "Negative Sampling Mode": neg_mode, "Cross Val Fold": CV, "Date": date, "Time": time})
        return pd.DataFrame(data)

    while True:
        os.system('clear')
        df = get_current_df()
        print("Model Training Progress Monitor")
        print("=====================================")
        print("Tracking:", output_folder)
        print("Last Update:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # Combine 'Date' and 'Time' columns into a single datetime column and find the earliest and latest timepoints
        datetimes = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        earliest_timepoint = datetimes.min()
        latest_timepoint = datetimes.max()
        time_difference = latest_timepoint - earliest_timepoint
        print("Approximate Running Time:", time_difference)
        print("=====================================")
        completed = len(df)
        total = len(model_combinations_df)
        print(f"Overall Progress: {completed}/{total} ({completed/total:.2%})")
        with tqdm(total=total, desc="Overall Progress") as pbar:
            pbar.update(completed)

        celltype_totals = model_combinations_df['Cell Type'].value_counts().to_dict()
        celltype_completions = df['Cell Type'].value_counts().to_dict()
        for celltype in celltype_totals:
            total = celltype_totals[celltype]
            completed = celltype_completions.get(celltype, 0)
            print(f"{celltype} Progress: {completed}/{total} ({completed/total:.2%})")
            with tqdm(total=total, desc=f"{celltype} Progress") as pbar:
                pbar.update(completed)
        time.sleep(60)