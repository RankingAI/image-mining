# Created by yuanpingzhou at 11/27/18

import pandas as pd
import numpy as np
import random

in_file = '../data/test/test_0819_part3.csv'
out_file_1 = '../data/test/test_0819_part3_1.csv'
out_file_2 = '../data/test/test_0819_part3_2.csv'
out_file_3 = '../data/test/test_0819_part3_3.csv'

n = 0
url_set = set()
head = ''
with open(in_file, 'r') as i_file,\
        open(out_file_1, 'w') as o_file_1, \
        open(out_file_2, 'w') as o_file_2, \
        open(out_file_3, 'w') as o_file_3:
    for line in i_file:
        if(n == 0):
            n += 1
            continue
        line = line.rstrip()
        parts = line.split(',')
        uid = parts[0]
        try:
            info_id = parts[1]
            url = parts[2]
            if(url not in url_set):
                url_set.add(url)
                proba = random.random()
                if(proba > 0.7):
                    o_file_1.write('{}\n'.format(line))
                elif(proba > 0.4):
                    o_file_2.write('{}\n'.format(line))
                else:
                    o_file_3.write('{}\n'.format(line))
        except:
            print(line)
        n += 1
o_file_1.close()
i_file.close()
