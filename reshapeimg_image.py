import numpy as np
import cv2
import os
from tqdm import tqdm


mtx =  np.array([[1579.0065371804494, 0.0, 1024.3801185727802], [0.0, 1697.5003831087758, 542.5693125986955],
                 [0.0, 0.0, 1.0]])
dist = np.array([[-0.6002237880679471], [0.36856632181945537], [-0.0018543577826724245], [-0.021979596228714066]
                 ,[-0.15631462404369234]])
K = np.array \
    ([[1065.9876632048952, 0.0, 970.0351475014528], [0.0, 1051.0552629200934, 533.2306815660671] ,[0.0, 0.0, 1.0]])
D = np.array([[0.09852134322204922], [0.0040876124771701385], [-0.14887184687293656], [0.07885494560173759]])



def calibration(in_dir, out_dir):
    file_list = os.listdir(in_dir)
    for filename in tqdm(file_list):
        oldname = os.path.join(in_dir, filename)
        img_old = cv2.imread(oldname)
        width, height = 1960, 1080
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
        mapx2, mapy2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, P, (width, height), cv2.CV_32F)
        frame_rectified = cv2.remap(img_old, mapx2, mapy2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("result", frame_rectified)
        cv2.imwrite(out_dir+"{}".format(filename), frame_rectified)



if __name__ == '__main__':
   in_dir = '/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/realtime-panorama-stitching/tree/'
   out_dir = '/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/realtime-panorama-stitching/'
   calibration(in_dir, out_dir)