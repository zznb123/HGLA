# ls = open('ChCh-Miner_durgbank-chem-chem.tsv').readlines()
# bio = ""
#
# for line in ls:
#     bio = bio + ",".join(line.split()) + ','+ '1'+ "\n"
# print(bio)
#
# f0 = open("ch_test2.csv", "x")
# f0.write(bio)
# f0.close()
import pandas as pd
# ch_i = pd.read_csv('ch.csv', names=['Drug1_ID','Drug2_ID'])
# print(ch_i.head())
# ch_i.to_csv('ch_i.csv', index=False)
ch_i = pd.read_csv('ch_i.csv')
temp_fold = ch_i.loc[ch_i["Drug1_ID"] == "DB01058"]
print(temp_fold)