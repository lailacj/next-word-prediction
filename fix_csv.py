import pandas as pd

file_path = '/users/diriho/data/diriho/next-word-prediction/data/james-michaelov_data/parsed_data/michaelov_2024.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Drop duplicates based on 'FullText', 'target_word', and 'cloz', keeping the first occurrence
df_unique = df.drop_duplicates(subset=['FullText', 'target_word', 'cloz']).copy()

# Reset 'sentence_num' to be sequential from 1
df_unique['sentence_num'] = range(1, len(df_unique) + 1)

# Save it back
df_unique.to_csv(file_path, index=False)
print(f"Original rows: {len(df)}, Unique rows: {len(df_unique)}")
