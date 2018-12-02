# Created by yuanpingzhou at 11/26/18

import pandas as pd
import numpy as np
import os,sys

import time
import urllib.request
import json
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

q = Queue()
curr = 0
total = 0
lock = threading.Lock()

def download(url):
    rsp = urllib.request.urlopen(url)
    return rsp.read()

def download_if_not_exists(url, p):
   if os.path.isfile(p):
       return
   else:
       with open(p, 'wb') as f:
           f.write(download(url))

def download_if_not_exists_multi_thread(index):
   global curr
   while not q.empty():
       try:
           begin = time.time()
           item = q.get()
           url = item[0]
           p = item[1]
           if os.path.isfile(p):
               continue
           else:
               with open(p, 'wb') as f:
                   f.write(download(url))
           end = time.time()
           lock.acquire()
           curr = curr + 1
           show_bar(curr, total)
           print('\nSucceed download {0} pics /{1}\nCost: {2}s'.format(curr, total, str(end - begin)))
           lock.release()
       except Exception as e:
           print(e)


def get_total_lines(f):
    i = 0
    with open(f) as lines:
        for line in lines:
            i += 1
    return i

def show_bar(curr, total, width=80):
    tmp = float(curr) /  float(total)
    percent = int(tmp * 100)
    s = "({0:3}%)".format(percent)
    bar_done = int(tmp * width)
    bar_undo = width - bar_done
    bar = '[' + '>' * bar_done + ' '* bar_undo + ']'
    clear_width = len(s) + len(bar) + 5
    sys.stdout.write(' ' * clear_width + '\r')
    sys.stdout.flush()
    sys.stdout.write('{0}{1}\r'.format(bar, s))
    sys.stdout.flush()

def read_excel(excel_file):
    ''''''
    df = pd.read_excel(excel_file, na_values= 'null', sheet_name= 'sheet1')
    df = df[['图片url', '类型']]
    df['image_url'] = df['图片url'].apply(lambda x: x.rstrip())
    df['label'] = df['类型'].apply(lambda x: x.rstrip())
    df.drop(['图片url', '类型'], axis= 1, inplace= True)
    df.loc[df['label'] == 'null', 'label'] = '正常'
    df.loc[df['label'] == '涉黄', 'label']  = '色情'

    print('unique url {}, total url {}'.format(len(np.unique(df['image_url'])), len(df)))

    return df

def load_image_label(input_files):
    ''''''
    print('\n')
    df_list = []
    for in_file in input_files:
        part_df = read_excel(in_file)
        df_list.append(part_df)
        print(in_file)
        print(part_df['label'].value_counts())
    merged_df = pd.concat(df_list, axis=0, ignore_index= True)
    print(merged_df['label'].value_counts())
    print('unique url {}, total url {}'.format(len(np.unique(merged_df['image_url'])), len(merged_df)))
    print('\n')
    image_labels = dict([(merged_df['image_url'][i], merged_df['label'][i]) for i in range(len(merged_df))])

    return image_labels

if __name__ == '__main__':
    ''''''
    #sexual_1109_file = '../data/labeled/性感review_1109.xlsx'
    #normal_1109_file = '../data/labeled/正常review_1109.xlsx'
    #toxic_1109_file = '../data/labeled/涉黄review_1109.xlsx'
    #toxic_0819_part2_file = '../data/labeled/涉黄review_0819_part2.xlsx'
    #input_files = [toxic_1109_file, normal_1109_file, sexual_1109_file, toxic_0819_part2_file]
    image_size = 300

    if(len(sys.argv) != 3):
        print('usage {}: input_files[file1,file2,...] output_dir')
        sys.exit(1)
    input_files = sys.argv[1].split(',')
    output_dir = sys.argv[2]

    # step 1: create output dirs
    labels = ['色情', '性感', '正常', 'unknown']
    for l in labels:
        if(os.path.exists('{}/{}'.format(output_dir, l)) == False):
            os.makedirs('{}/{}'.format(output_dir, l))

    # step 2: load image labels
    image_labels = load_image_label(input_files)
    print('total images {}'.format(len(image_labels)))
    for l in labels:
        s = [url for url in image_labels if(image_labels[url] == l)]
        print('label {} count {}'.format(l, len(s)))

    #sys.exit(1)
    # step 3: fill the shared queue
    for url in image_labels:
        image_name = '{}/{}/{}'.format(output_dir, image_labels[url], url.split('/')[-1])
        if os.path.isfile(image_name):
            if os.path.getsize(image_name) != 0.0:
                continue
            elif os.path.getsize(image_name) == 0.0:
                os.remove(image_name)
        url = '{}?w={}'.format(url, image_size)
        q.put((url, image_name))
        total += 1
    print('downloading images {}'.format(total))

    # step 4: download images
    t_pool = ThreadPoolExecutor()
    for j in range(150):
        t_pool.submit(download_if_not_exists_multi_thread, j)
    t_pool.shutdown()
