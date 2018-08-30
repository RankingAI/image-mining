import config
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

import matplotlib.pyplot as plt

from skimage.io import imread, imsave

data_source = '0819'
level = 'toxic'
PredictOutputFile = '%s/%s.csv' % (config.TestOutputDir, data_source)

image_files = []
truth_label = []
predict_proba = []
with open(PredictOutputFile, 'r') as i_file:
    for line in i_file:
        line = line.rstrip()
        if (not line):
            continue
        parts = line.split(',')
        label = int(parts[1])
        predict = float(parts[2])
        truth_label.append(label)
        predict_proba.append(predict)
        image_files.append(parts[0])
i_file.close()
truth_label = [1 if (v >= config.level_label_dict[level]) else 0 for v in truth_label]

predict_toxic = [1 if (v > 0.55) else 0 for v in predict_proba]
print('precision %.6f, recall %.6f' % (precision_score(truth_label, predict_toxic), recall_score(truth_label, predict_toxic)))
print('predict positive %s' % np.sum(predict_toxic))

# save error
error_index = np.argwhere(np.array(truth_label) != np.array(predict_toxic)).flatten()
with open('%s/error_%s.txt' % (config.TestOutputDir, data_source), 'w') as o_file:
    for err_idx in error_index:
        o_file.write('%s,%s,%.6f\n' % (image_files[err_idx], truth_label[err_idx], predict_proba[err_idx]))
o_file.close()

# visualize error images
num_image = 50
num_columns = 10
sampled_index = np.random.choice(error_index, num_image)
plt.figure(figsize=(30,30))
plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.1)
i = 0
for err_idx in sampled_index:
    img = imread(image_files[err_idx])
    plt.subplot(int(num_image/num_columns), num_columns, i + 1)
    plt.imshow(img)
    title = 'image %s \ntruth %s, predict %.6f' % ('/'.join(image_files[err_idx].split('/')[-2:]), truth_label[err_idx], predict_proba[err_idx])
    print(title)
    plt.title(title)
    i += 1
plt.tight_layout()
plt.savefig('error.jpg')
