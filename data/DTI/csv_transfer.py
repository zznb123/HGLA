ls = open('biosnap.tsv').readlines()
bio = ""
for line in ls:
    bio = bio + ",".join(line.split()) + "\n"
print(bio)

f0 = open("bios.csv", "x")
f0.write(bio)
f0.close()