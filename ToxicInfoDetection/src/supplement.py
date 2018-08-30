import glob
import config
import os,sys
from skimage.io import imread, imsave
import numpy as np

supplement_rate = 0.2

def supplement(test_dir, cate):
    ''''''
    ## load images with certain category
    remained_images = []
    cate_image_files = []
    for f in glob.glob('%s/*/*/*.jpg' % test_dir):
        parts = f.split('/')
        level = config.level_zn_en[parts[-3]]
        cate_id = int(parts[-2])
        if(cate_id == cate):
            cate_image_files.append(f)
        else:
            remained_images.append(f)
    ## sample
    sampled_image_files = []
    for i in range(len(cate_image_files)):
        if(np.random.random() < supplement_rate):
            sampled_image_files.append(cate_image_files[i])
        else:
            remained_images.append(cate_image_files[i])
    ## save the remained
    num_remained_err = 0
    cnt = 0
    for im_file in remained_images:
        try:
            img = imread(im_file)
            parts = im_file.split('/')
            new_remained_dir = '%s_new/%s/%s' % (config.test_data_set['0819'], parts[-3], parts[-2])
            if(os.path.exists(new_remained_dir) == False):
                os.makedirs(new_remained_dir)
            new_im_file = '%s/%s' % (new_remained_dir, parts[-1])
            imsave(new_im_file, img)
            if(cnt % 200 == 0):
                print('%s done.' % cnt)
        except:
            num_remained_err += 1
        cnt += 1
    print('saving the remained done.')
    ## save the sampled
    num_sampled_err = 0
    cnt = 0
    for im_file in sampled_image_files:
        try:
            img = imread(im_file)
            parts = im_file.split('/')
            img_name = parts[-1].split('.')[0]
            cate_id = int(parts[-2])
            level = parts[-3]
            new_supplement_dir = '%s_new/%s' % (config.test_data_set['history'], level)
            if(os.path.exists(new_supplement_dir) == False):
                os.makedirs(new_supplement_dir)
            new_img_file = '%s/%s-%s.jpg' % (new_supplement_dir, cate_id, img_name)
            imsave(new_img_file, img)
            if(cnt % 200 == 0):
                print('%s done.' % cnt)
        except:
            num_sampled_err += 1
        cnt += 1

    print('saving for the remained err %s, sampled %s' % (num_remained_err, num_sampled_err))
    return sampled_image_files

if __name__ == '__main__':
    ''''''
    test_image_dir = config.test_data_set['0819']
    supplement(test_image_dir, 101)
