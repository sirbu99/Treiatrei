import matplotlib.pyplot as plt
import numpy as np

#create datasets
entriesBayes=[]
entriesRandom=[]

#precision	recall
with open('logs/BayNv.log', 'r') as f:
    lines=f.readlines()
for line in lines:
    if line[0] == '#':
        pass
    else:
        values=line.split(sep="\t")
        entriesBayes.append(values)

with open('logs/Flip.log', 'r') as f:
    lines=f.readlines()
for line in lines:
    if line[0] == '#':
        pass
    else:
        values=line.split(sep="\t")
        entriesRandom.append(values)

#we do the average
precisionBayes=0
precisonRand=0
recallBayes=0
recallRand=0
for i in range(len(entriesBayes)):
    precisionBayes+=float(entriesBayes[i][0])
    precisonRand+=float(entriesRandom[i][0])
    recallBayes+=float(entriesBayes[i][1])
    recallRand+=float(entriesRandom[i][1])
precisionBayes=precisionBayes/len(entriesBayes)
precisonRand=precisonRand/len(entriesRandom)
recallBayes=recallBayes/len(entriesBayes)
recallRand=recallRand/len(entriesRandom)

#creating final dataset
height=[precisionBayes,recallBayes,precisonRand,recallRand]

#creating bars
bars=["BN Avg Precision", "BN Avg Recall", "CF Avg Precision","CF Avg recall"]
x_pos = np.arange(len(bars))
plt.bar(x_pos, height, color=['green','green','red','red'])
# Create names on the x-axis
plt.xticks(x_pos, bars)
# Show graph
plt.show()
    