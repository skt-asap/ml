import pandas as pd

file_path = '../ELG_Busan_PoC_per_CA_site_0226_0519.csv'
df = pd.read_csv(file_path)

print(df.dtypes)