#-*- coding:utf-8 -*-
import sys
import os
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

if __name__ == '__main__':
    ## declaration
    data_root = '../data'
    pic_prefix = 'http://pic1.zhuanstatic.com/zhuanzh'
    output_path = "../data/raw/unlabeled/image"
    url_file = '{}/unlabeled.txt'.format(data_root)

    image_size = 300

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ##  step 1: get all image files
    meta_data = []
    n = 0
    with open(url_file, 'r', encoding= 'utf-8') as o_file:
        for line in o_file:
            if(n == 0):
                n += 1
                continue
            line = line.strip().split("\t")
            uid = line[0]
            info_id = line[1]
            image_prefix = '{}_{}'.format(uid, info_id)
            pic_names = line[2].split('|')
            for pic in pic_names:
                meta_data.append({'image_prefix': image_prefix, 'image_name': pic})
            n += 1
    o_file.close()

    print('\n================================')
    print('total image files {}'.format(len(meta_data)))
    print('================================\n')

    ## step 2: preprocessing 
    err = 0
    urls_count = len(meta_data)
    total = urls_count
    for item in meta_data:
        if((item['image_name'].endswith('jpg') == False) & (item['image_name'].endswith('jpeg') == False) & (item['image_name'].endswith('png') == False)):
            #print(item['image_name'])
            err += 1
            continue
        suffix = item['image_name'].split('.')[-1]
        if(item['image_name'].startswith('http')):
            url = '{}?w={}'.format(item['image_name'], image_size)
            image_name = '{}_{}'.format(item['image_prefix'], item['image_name'].split('/')[-1])
        elif(item['image_name'].startswith('n_')):
            url = '{}/{}?w={}'.format(pic_prefix, item['image_name'], image_size)
            image_name = '{}_{}'.format(item['image_prefix'], item['image_name'])
        else:
            #print(item['image_name'])
            err += 1
            continue
        file_name = os.path.join(output_path, '{}'.format(image_name))
        if os.path.isfile(file_name):
            if os.path.getsize(file_name) != 0.0:
                continue
            elif os.path.getsize(file_name) == 0.0:
                os.remove(file_name)
        q.put((url,file_name))

    print('\n==============================')
    print('!!!! Error image file number {}'.format(err))
    print('===============================\n')

    ## step 3: download images
    t_pool = ThreadPoolExecutor()
    for j in range(150):
        t_pool.submit(download_if_not_exists_multi_thread, j)
    t_pool.shutdown()
