# Created by yuanpingzhou at 12/3/18

import ToxicImageDetection_pb2
import ToxicImageDetection_pb2_grpc
from threading import Thread
import grpc
from grpc.beta import implementations
import numpy as np
import time
import utils

num_threads = 2 
num_times = 1 
num_images = 1 

#server_ip = '10.135.9.3'
server_ip = '127.0.0.1'
server_port = 50051
timeout = 60 
results = np.full((num_threads, num_times), 0.0)
error = np.full(num_threads, 0.0)

class TextRequest(Thread):
    def __init__(self, no):
        super().__init__()
        ''''''
        self._no = no
        # open a gRPC channel
        self.channel = grpc.insecure_channel('{}:50051'.format(server_ip))
        # create a stub (client)
        self.stub = ToxicImageDetection_pb2_grpc.NSFWStub(self.channel)

    def run(self):
        ''''''
        for i in range(num_times):
            test_urls = [
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v29dcd20fa6d534c00a083943f7d0005fb.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2387fc96f7a6b479e966f499b4dcba21c.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2b869333c58fe497e91f75fe362facd3b.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v218c13be2bebf4a03982afce48a0b9b65.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v239a3ab34023644a8bbf3fb6b3085ba64.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2d5f2b74b87854442a2cfb42196f96a95.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2f436745f42104d46a0b4d292718c824c.jpg',
                         # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2fc9d14b940934773bd07607b598734af.jpg'
                # 'http://pic1.zhuanstatic.com/zhuanzh/n_v2504121b4ea614da4956327d359612a5a.jpg',
                # 'http://pic1.zhuanstatic.com/zhuanzh/n_v27aa487c0189e49f5b3b5021fdc8d9862.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v288491acfda06417a972787747e8a6346.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v2480e4e8d21154c4d98b5cb68251d3349.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v29842134455ee4545a430e72255ef0d16.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v213b5b4961778484d9ddfcaedf18c27bc.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v2d69d2335745b4500b2f9443e633d300a.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v20555a9ed6faa457e8b8744cac8bb88cd.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v24462d5c076214732b21c071050db3250.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v2e348efcb85f54d21af81592d706cb562.jpg',
                'http://pic1.zhuanstatic.com/zhuanzh/n_v229f63a1c69b243459e7a51080566d7c5.jpg'
                         ] * num_images
            request = ToxicImageDetection_pb2.ImageURL()
            request.urls.extend(test_urls)
            try:
                t1 = time.time()
                response = self.stub.OpenNSFW(request)
                print(response)
                t2 = time.time()
                results[self._no][i] = t2 - t1
            except Exception as e:
                error[self._no] += 1
                print(e)
                continue

threads = [TextRequest(i) for i in range(num_threads)]
start = time.time()
with utils.timer('%s REQUEST' % num_threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print('Error analysis:')
    print(np.sum([1 for i in range(len(error)) if(error[i] > 0)]), num_threads)
    print('Time analysis:')
    print('mean time cost ')
    print(results.mean(axis= 1))
    print('maximum time cost ')
    print(results.max(axis= 1))
    print('minimum time cost')
    print(results.min(axis= 1))

    #print('success %s/%s ' % (len(np.where(results > 0.5)[0]), num_threads))
end = time.time()
print('time %s' % (end - start))