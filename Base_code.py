import pandas as pd

df = pd.read_csv("DAA01.20250205091109.csv")

#Dropping 'State' rows as they are just the sum of other columns
condition = df['Regional Veterinary Offices'] == 'State'
df = df[~condition]

#Anonymising the office names
anonymization_map = {office: f'Office {i+1}' for i, office in enumerate(df['Regional Veterinary Offices'].unique())}
df['Regional Veterinary Offices'] = df['Regional Veterinary Offices'].map(anonymization_map)

#Expanding the dataset horizontally for making it suitable for synthetic data generation
base_df = df[['Year', 'Regional Veterinary Offices', 'C03321V04008']].drop_duplicates().set_index('Year')
data = pd.DataFrame()
statistics = df['STATISTIC'].unique()

for statistic in statistics:
    temp_df = df[df['STATISTIC'] == statistic].set_index('Year')[['VALUE']]
    temp_df.columns = [statistic]
    if data.empty:
        data = temp_df
    else: 
        data = pd.concat([data, temp_df], axis=1)

result = pd.concat([base_df, data], axis=1)

result.to_csv("Dataset_v2.csv")


