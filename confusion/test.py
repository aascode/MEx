import csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

reader = csv.reader(open('/Users/anjanawijekoon/IdeaProjects/MEx/confusion/cnn_dc.csv', "r"), delimiter=",")
data = []
for row in reader:
    data.append(row)

all_data = {0:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            1:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            2:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            3:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            4:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            5:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
            6:{0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}

for i in range(30):
    all = data[10*i:10*(i+1)]
    all = all[1:]
    predicted = all[0][0].replace('Predicted ', '').strip().split(' ')
    predicted = [int(x) for x in predicted if x]
    for item in all[2:]:
        actual = item[0]
        actual = actual.split(' ')
        actual = [int(x) for x in actual if x]
        actual_label = actual[0]
        if len(actual[1:]) == len(predicted):
            for index, ps in enumerate(actual[1:]):
                all_data[actual_label][predicted[index]] += ps

actuals = []
for actual in all_data:
    preds = []
    predications = all_data[actual]
    for p in predications:
        preds.append(predications[p])
    actuals.append(preds)
    print(','.join(str(t) for t in preds))


df_cm = pd.DataFrame(actuals, index = [i for i in range(7)], columns = [i for i in range(7)])
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm, annot=True)
plt.show(sn)