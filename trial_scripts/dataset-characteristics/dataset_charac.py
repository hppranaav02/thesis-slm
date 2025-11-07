import json
import pandas as pd
import matplotlib.pyplot as plt

records = []
with open('/local/s3905020/code/dataset-creation/train.jsonl', 'r') as f:
    for line in f:
        obj = json.loads(line)
        records.append({
            'code': obj['input'],
            'cwes': obj['output']
        })
df = pd.DataFrame(records)

df['is_secure'] = df['cwes'].apply(lambda x: len(x) == 0)
total = len(df)
secure_count = df['is_secure'].sum()
insecure_count = total - secure_count
print(f"Total snippets: {total}")
print(f"Secure: {secure_count} ({secure_count/total:.1%})")
print(f"Insecure: {insecure_count} ({insecure_count/total:.1%})")

df['num_cwes'] = df['cwes'].apply(len)
# print("CWE count stats:")
# print(df['num_cwes'].describe())

"""
List of all CWEs in the dataset and number of snippets containing each CWE
"""
cwe_counts = df['cwes'].explode().value_counts()
print("CWE counts:")
for cwe, count in cwe_counts.items():
    print(f"{cwe}: {count} snippets")

plt.figure(figsize=(10, 6))
df['num_cwes'].hist(bins=range(df['num_cwes'].max()+2), edgecolor='black')
plt.title('Distribution of CWE Counts per Snippet')
plt.xlabel('Number of CWEs')
plt.ylabel('Number of Snippets')
plt.xticks(range(df['num_cwes'].max()+1))
plt.grid(axis='y')
plt.show()

# plt.figure()
# df['num_cwes'].hist(bins=range(df['num_cwes'].max()+2))
# plt.title('Distribution of CWE Counts per Snippet')
# plt.xlabel('Number of CWEs')
# plt.ylabel('Number of Snippets')
# plt.show()

from collections import Counter
all_cwes = [cwe for sublist in df['cwes'] for cwe in sublist]
cwe_counts = Counter(all_cwes)
top10 = cwe_counts.most_common(10)
print("Top 10 CWE IDs:")
for cwe, cnt in top10:
    print(f"  {cwe}: {cnt} occurrences")

# df['loc'] = df['code'].apply(lambda s: len(s.splitlines()))
# corr = df[['loc', 'num_cwes']].corr().iloc[0,1]
# print(f"Correlation between LOC and CWE count: {corr:.2f}")

# plt.figure()
# plt.scatter(df['loc'], df['num_cwes'], alpha=0.3)
# plt.title('LOC vs. CWE Count')
# plt.xlabel('Lines of Code')
# plt.ylabel('Number of CWEs')
# plt.show()
