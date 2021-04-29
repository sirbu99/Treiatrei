import matplotlib.pyplot as plt
import numpy as np
import os


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
    bars.append(key[:-6] + " f-measure")
    # bars.append(key[:-6] + " recall")
print(bars)
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

ax.barh(y_pos, height, color=colors, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(bars)
ax.invert_yaxis()  # labels read top-to-bottom
fig.subplots_adjust(left=0.5)
ax.set_xlabel('Performance')
plt.show()
