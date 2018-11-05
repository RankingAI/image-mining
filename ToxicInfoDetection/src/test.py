import config
import zz_nsfw
#from sklearn.metrics import precision_score, recall_score, accuracy_score

truth = []
predict = []
test_log_file = '%s/%s/test/test_log.txt' % (config.ModelRootDir, config.strategy)
with open(test_log_file, 'r') as i_file:
    for line in i_file:
        line = line.rstrip()
        if(not line):
            continue
        parts = line.split(',')
        y = int(parts[1])
        predict_label =  int(parts[2])
        truth.append(y)
        predict.append(predict_label)
i_file.close()

# evaluation
print('\n ======= Summary ========')
for l in ['toxic', 'sexual', 'normal']:
    num_true_pos, num_pred_pos, accuracy, precision, recall = zz_nsfw.zz_metric(y, predict, l)
    print('%s on Test: accuracy %.6f, truth positve %s, predict positive %s, precision %.6f, recall %.6f.' %
          (l, accuracy, num_true_pos, num_pred_pos, precision, recall))
print('==========================\n')

