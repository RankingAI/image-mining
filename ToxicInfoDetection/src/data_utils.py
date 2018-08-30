import glob
import numpy as np

import config

def load_files(input_dir, mode, sample_rate= 0.02):
    ''''''
    print('loading directory %s' % input_dir)
    image_files = []
    labels = []
    for en_l in config.level_label_dict.keys():
        zn_l = config.level_en_zn[en_l]
        if((mode == 'history') | (mode == 'history_supplement')):
            expr_files = '%s/%s/*.jpg' % (input_dir, zn_l)
            print('load %s' % expr_files)
            files = glob.glob(expr_files)
        elif((mode == '0819') | (mode == '0819_new')):
            expr_files = '%s/%s/*/*.jpg' % (input_dir, zn_l)
            print('load %s' % expr_files)
            files = glob.glob(expr_files)
        else:
            files = []
        image_files.extend(files)
        labels.extend([config.level_label_dict[en_l]] * len(files))
    image_files = np.array(image_files)
    labels = np.array(labels)
    print('total image files %s' % len(image_files))
    # sampling
    if((sample_rate >= 0.0) & (sample_rate < 1.0)):
        sample_index = np.random.choice(len(image_files), int(sample_rate * len(image_files)))
        image_files = image_files[sample_index]
        labels = labels[sample_index]
        print('sampled image files %s' % (len(image_files)))

    return image_files, labels

def save_predictions(image_files, truth_label, pred_scores, output_file):
    ''''''
    with open(output_file, 'w') as o_file:
        for i in range(len(image_files)):
            o_file.write('%s,%s,%.6f' % (image_files[i], truth_label[i], pred_scores[i]))
    o_file.close()
