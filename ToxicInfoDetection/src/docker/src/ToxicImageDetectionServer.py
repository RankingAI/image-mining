# Created by yuanpingzhou at 12/3/18

import grpc
from concurrent import futures
import time
import argparse
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler

from OpenNSFWWrapper import ModelFactory, ImageClassifyService
#from OpenNSFWWrapper_1 import ImageClassifyService
import ToxicImageDetection_pb2_grpc

#datestr = datetime.datetime.now().strftime("%Y%m%d")
if __name__ == '__main__':
    ''''''
    current_dir = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument('-nsfw_model_dir', "--nsfw_model_dir",
                        help="directory of nsfw saved model",
                        default='{}/../model/nsfw/savedmodel'.format(current_dir),
                        )

    parser.add_argument("-nsfw_model_version", "--nsfw_model_version",
                        help="version of nsfw model",
                        default= '1'
                        )

    parser.add_argument("-zz_nsfw_model_dir", "--zz_nsfw_model_dir",
                        help="directory of zz_nsfw saved model",
                        default='{}/../model/zz_nsfw/infer'.format(current_dir),
                        )

    parser.add_argument("-zz_nsfw_model_version", "--zz_nsfw_model_version",
                        help="version of zz_nsfw model",
                        default= '1'
                        )

    parser.add_argument('-num_zz_nsfw_model', '--num_zz_nsfw_model',
                        help= 'number of zz_nsfw',
                        default= '5'
    )

    parser.add_argument('-num_class', '--num_class',
                        help= 'number of classes',
                        default= '3'
                        )

    parser.add_argument('-num_worker', '--num_worker',
                        help= 'number of workers',
                        default= '50'
                        )

    parser.add_argument('-image_size', '--image_size',
                        help= 'size of image',
                        default= '300'
                        )

    parser.add_argument('-log_dir', '--log_dir',
                        help= 'log dir',
                        default= '{}/log'.format(current_dir)
    )
    args = parser.parse_args()

    if(os.path.exists(args.log_dir) == False):
        os.makedirs(args.log_dir)

    log_formatter = logging.Formatter('%(asctime)s %(funcName)s:%(lineno)d %(message)s')
    my_handler = RotatingFileHandler('{}/{:%Y-%m-%d}.log'.format(args.log_dir, datetime.now()),
                                     mode='a', maxBytes= 200 * 1024 * 1024,
                                     backupCount= 20, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)

    app_log = logging.getLogger('')
    app_log.setLevel(logging.DEBUG)
    app_log.addHandler(my_handler)

    nsfw_saved_model_dir = args.nsfw_model_dir
    nsfw_model_version = int(args.nsfw_model_version)
    zz_nsfw_saved_model_dir = args.zz_nsfw_model_dir
    zz_nsfw_model_version = int(args.zz_nsfw_model_version)
    num_zz_nsfw_model = int(args.num_zz_nsfw_model)
    num_class = int(args.num_class)
    num_worker = int(args.num_worker)
    image_size = int(args.image_size)

    logging.debug(' \t \t TOXIC IMAGE DETECTION SERVICE PARAMETERS \t \t')
    logging.debug('parameters: nsfw_saved_model_dir {}, nsfw_model_version {},'
                  'zz_nsfw_saved_model_dir {}, zz_nsfw_model_version {}, '
                  'num_zz_nsfw_model {}, num_class {}, num_worker {}, image_size {}'.format(
        nsfw_saved_model_dir, nsfw_model_version, zz_nsfw_saved_model_dir, zz_nsfw_model_version,
        num_zz_nsfw_model, num_class, num_worker, image_size
    ))

    # step 1: load models
    try:
        ModelFactory.initial_models(num_zz_nsfw_model, nsfw_saved_model_dir, nsfw_model_version, zz_nsfw_saved_model_dir, zz_nsfw_model_version)
    except Exception as e:
        logging.debug('Error: initial models failed')
        sys.exit(1)

    # step 2: create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers= num_worker))

    # step 3: register response function
    ToxicImageDetection_pb2_grpc.add_NSFWServicer_to_server(ImageClassifyService(
        #nsfw_saved_model_dir,
        #nsfw_model_version,
        #zz_nsfw_saved_model_dir,
        #zz_nsfw_model_version,
        num_class= num_class,
        num_zz_nsfw_model= num_zz_nsfw_model,
        image_size= image_size
    ), server)

    # step 4: open port, and launch server
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.debug('Starting server, Listening on port 50051 ...')

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
