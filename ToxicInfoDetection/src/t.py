# Created by yuanpingzhou at 11/2/18

import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score

y = tf.placeholder(tf.int32, shape= [None, ])

proba = tf.placeholder(tf.float32, shape= [None, 3])

idx = 1

labels = tf.equal(y, tf.cast(idx, tf.int32))
predicts = tf.equal(tf.cast(tf.argmax(proba, 1), tf.int32), tf.cast(idx, tf.int32))
recall = tf.metrics.recall(labels= labels, predictions= predicts)

#recall = tf.metrics.recall(labels= y, predictions= tf.cast(tf.argmax(proba, 1), tf.int32))

data_proba = [[0.6, 0.4, 0.0], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]]
t = np.argmax(data_proba, 1)
data_y = [1, 1, 0, 2]

print(t)
print(data_y)
#print(recall_score(data_y, t))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    ret = sess.run(recall, feed_dict= {y: data_y, proba: data_proba})
    #ret = sess.run(predicts, feed_dict= {y: data_y, proba: data_proba})
    print(ret)
