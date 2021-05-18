import matplotlib.pyplot as plt
import numpy as np
import os

LOGGING = 0
def getList(dict):
    return dict.keys()


def getList(dict):
    return dict.keys()


def validLine(line):
    if line[0] != '#':
        return True
    return False


names = []
d = {}

for filename in os.listdir('logs'):
    if filename.endswith(".log"):
        d[filename[:-4]] = None
# file format:
# precision	recall
for filename in os.listdir('logs'):
    if filename.endswith(".log"):
        with open('logs/' + filename, 'r') as f:
            # work with the current file, make the average then put a tuple it in our dict
            name = filename[:-4]
            lines = f.readlines()
            values = []
            for line in lines:
                if validLine(line):
                    values += line.split(sep="\t")
                else:
                    pass
            current = "precision"
            precision = []
            precisionc = 0
            recall = []
            recallc = 0
            for i in values:
                if current == "precision":
                    precisionc += 1
                    precision.append(float(i))
                    current = "recall"
                if current == "recall":
                    recallc += 1
                    recall.append(float(i))
                    current = "precision"
            t = []
            t = [sum(precision) / precisionc, sum(recall) / recallc]
            d[name] = t
newd = {k: v for k, v in d.items() if v is not None}
names = getList(d)
bars = []
for key in newd:
    bars.append(key[:-6] + " precision")
    bars.append(key[:-6] + " recall")
# print(bars)
y_pos = np.arange(len(bars))
colors = []
height = []
fig, ax = plt.subplots()
for key in newd:
    t = newd[key]
    height.append(t[0])
    height.append(t[1])
    if t[0] > 0.85:
        colors.append("green")
    else:
        colors.append("red")
    if t[1] > 0.85:
        colors.append("green")
    else:
        colors.append("red")

names = []
d = {}

for filename in os.listdir('logs'):
    if filename.endswith(".log"):
        d[filename[:-4]] = None
# file format:
# precision	recall
for filename in os.listdir('logs'):
    if filename.endswith(".log"):
        with open('logs/' + filename, 'r') as f:
            # work with the current file, make the average then put a tuple it in our dict
            name = filename[:-4]
            lines = f.readlines()
            values = []
            for line in lines:
                if validLine(line):
                    values += line.split(sep="\t")
                else:
                    pass
            fMeasure = []
            fMeasurec = 0
            for i in values:
                fMeasurec += 1
                fMeasure.append(float(i))
            t = []
            t = [sum(fMeasure) / fMeasurec]
            d[name] = t
newd = {k: v for k, v in d.items() if v is not None}
names = getList(d)
bars = []
for key in newd:
    bars.append(key[:-6])
    # bars.append(key[:-6] + " recall")
# print(bars)
y_pos = np.arange(len(bars))
colors = []
height = []
fig, ax = plt.subplots()
for key in newd:
    t = newd[key]
    height.append(t[0])
    if t[0] > 0.55:
        colors.append("green")
    else:
        colors.append("red")
if LOGGING == 1:
    with open('stats_for_hybrid.txt', "a+") as f:
        index = bars.index("BayNvLem")
        f.write(str(d["BayNvLemGroups"][0])+"\n")
        f.write(str(d["FlipLemGroups"][0])+"\n")
        f.write(str(d["AdaBoostLemGroups"][0])+"\n")
ax.barh(y_pos, height, color=colors, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(bars)
ax.invert_yaxis()  # labels read top-to-bottom
fig.subplots_adjust(left=0.5)
ax.set_xlabel('F-measure')
plt.show()
