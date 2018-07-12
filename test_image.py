import cv2
from dface.core.detect import create_mtcnn_net, MtcnnDetector
import dface.core.vision as vision
import glob

image_path = './'
save_path = './save'

# Get user supplied values
file_names = glob.glob(image_path + '/*.png') + glob.glob(image_path + '/*.jpg')

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt", r_model_path="./model_store/rnet_epoch.pt", o_model_path="./model_store/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    for f in file_names:
        print(f)

        smilePath = 'haarcascade_smile.xml'
        smileCascade = cv2.CascadeClassifier(smilePath)
        
        img = cv2.imread(f)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #b, g, r = cv2.split(img)
        #img2 = cv2.merge([r, g, b])

        bboxs, landmarks = mtcnn_detector.detect_face(img)

        height, width, channels = img.shape

#        print(bboxs)
# if len(bboxs) != 0:
        x1 = bboxs[0][0]
        y1 = bboxs[0][1]
        x2 = bboxs[0][2]
        y2 = bboxs[0][3]
        w = (x2 - x1)
        h = (y2 - y1)
        
#        print(w/2+x1,h/2+y1)
#        print(h/height, w/width)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        smile = smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.16,
            minNeighbors=35,
            minSize=(15, 15),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        name = f.split('\\')[0]
        file1 = open('./save/%s.txt' % name.split('.')[1], 'wt')
        print(smile)
        if len(smile) == 0:
            file1.write('0 %06f %06f %06f %06f \n' %((w/2+x1)/width, (h/2+y1)/height, w/width, h/height))
        else:
            file1.write('1 %06f %06f %06f %06f \n' %((w/2+x1)/width, (h/2+y1)/height, w/width, h/height))
        
#        vision.vis_face(img_bg,bboxs,landmarks)
        file1.close()
