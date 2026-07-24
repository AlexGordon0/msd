import pandas as pd

pd.set_option("display.max_rows", None)

# List of your filenames
files = [
    "data/baseline_results.csv",
    # "data/lstm_2d_results_conf.csv",
    "data/combined_results.csv",
]


def merge_csv_on_video(file_list):
    columns = ["filename", "total_speech", "confusion"]
    # Load the first dataframe
    df_final = pd.read_csv(file_list[0])[columns]

    # Rename columns (except the join key) to identify the source
    df_final = df_final.rename(
        columns={col: f"{col}_v1" for col in df_final.columns if col != "filename"}
    )

    # Loop through the remaining files
    for i, file in enumerate(file_list[1:], start=2):
        df_temp = pd.read_csv(file)[columns]

        # Rename columns for source i
        df_temp = df_temp.rename(
            columns={col: f"{col}_v{i}" for col in df_temp.columns if col != "filename"}
        )

        # Merge on 'filename' (the unique video ID)
        df_final = pd.merge(df_final, df_temp, on="filename", how="outer")

    return df_final


# Execute the merge
merged_df = merge_csv_on_video(files)
# diff1_3 = merged_df["confusion_v1"] - merged_df["confusion_v3"]
# diff2_3 = merged_df["confusion_v2"] - merged_df["confusion_v3"]
# merged_df["min_der_diff"] = merged_df["confusion_v1"] - merged_df["confusion_v2"]

# 4. Create the new column using the minimum of the two differences
# merged_df["min_der_diff"] = pd.concat([diff1_3, diff2_3], axis=1).min(axis=1)

filename_list = [
    "c9f96W4TOXQ_1",
    "mulxVBOq-S8_0",
    "vhu3LxxG-Tg_0",
    "Q3roIYzQSGo_1",
    "3sNqcTzdWMU_1",
    "zWpoSvLmrW4_0",
    "pVYn7eRNTCg_1",
    "iWE-vOq4Cj8_0",
    "lNVVNxtolyE_1",
    "h0HjktaeEpA_0",
]

merged_df["confusion_v1"] = (
    merged_df["confusion_v1"] / merged_df["total_speech_v1"] * 100
)
merged_df["confusion_v2"] = (
    merged_df["confusion_v2"] / merged_df["total_speech_v1"] * 100
)
merged_df = merged_df[["filename", "confusion_v1", "confusion_v2"]]
merged_df = merged_df[merged_df["filename"].isin(filename_list)]
print(merged_df)

# Optional: Save the result
# merged_df.to_csv('merged_results.csv', index=False)
