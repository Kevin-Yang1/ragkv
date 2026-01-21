import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import argparse
import os

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def main(FOLDER_PATH, reuse_method,max_capacity_prompt,use_calibration,PRETRAINED_LEN=8192):
    # Path to the directory containing JSON results
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FOLDER_PATH) + os.sep

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    
    if json_files.__len__() == 0: 
        raise ValueError("No json files found in the directory")
    

    # List to hold the data
    data = []
    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        # import pdb; pdb.set_trace()
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))

            # if "eat a sandwich and sit in Dolores Park on a sunny day" in model_response:
            #     score = 1
            # else:
            #     score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))

            # Appending to the list
            data.append({
                "Document_Depth": document_depth,
                "Context_Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    locations = list(df["Context_Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): 
            break
    pretrained_len = li
    # import pdb; pdb.set_trace()
    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document_Depth', 'Context_Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document_Depth", columns="Context_Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(14, 8))  # Can adjust these dimensions as needed
    # f = plt.figure(figsize=(14, 8))
    
    # columns_to_keep = [pivot_table.columns[0]]
    # columns_to_keep.extend(pivot_table.columns[i] for i in range(1, len(pivot_table.columns) - 1, 2))
    # columns_to_keep.append(pivot_table.columns[-1])
    # pivot_table_filtered = pivot_table[columns_to_keep]

    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score', 'pad': 0.01},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )
    
    # 获取x轴的刻度位置和标签
    xticks = heatmap.get_xticks()
    xticklabels = heatmap.get_xticklabels()

    # 只保留能被200整除的刻度标签
    new_xticks = []
    new_xticklabels = []
    for i, col in enumerate(pivot_table.columns):
        try:
            val = int(col)
            if val % 512 == 0 or val == 8000:
                new_xticks.append(xticks[i])
                new_xticklabels.append(str(val))
        except:
            continue
    # new_xticks = [tick for i, tick in enumerate(xticks) if int(pivot_table.columns[i]) % 512 == 0]
    # new_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if int(pivot_table.columns[i]) % 512 == 0]

    # 设置新的x轴刻度和标签
    heatmap.set_xticks(new_xticks)
    heatmap.set_xticklabels(new_xticklabels, rotation=45, fontsize=23)
    
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Score', fontsize=30)
    cbar.ax.tick_params(labelsize=25)

    method='Ours'

    # is_cali = "w CaliDrop" if 'true' in use_calibration else "w/o CaliDrop"

    average_score = df["Score"].mean() * 100
    average_score_formatted = round(average_score, 1)
    
    plt.title(f'{method}  Acc {average_score_formatted}', fontsize=30)  # Adds a title
    plt.xlabel('Token Limit', fontsize=30)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=30)  # Y-axis label
    plt.xticks(rotation=45, fontsize=23)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=25)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    # plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)


    save_path = f'./needle_figs/{reuse_method}.pdf'
    print("saving at %s" % save_path)
    plt.savefig(save_path)


if __name__ == "__main__":

    reuse_method = 'debug':
    result_dir = f'./outputs/Mistral-7B-Instruct/{reuse_method}/needle'
    # result_dir = f'./outputs/Qwen2.5-7B-Instruct/{reuse_method}/needle'
    # result_dir = f'./outputs/Llama-3-8B-Instruct/{reuse_method}/needle'
    max_capacity_prompt=1000
    use_calibration='False'

    main(result_dir, reuse_method, max_capacity_prompt, use_calibration)
