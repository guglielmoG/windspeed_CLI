import numpy as np
import cv2
import os
import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from libs.retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from libs.retinanet.keras_retinanet import models
import matplotlib.pyplot as plt
from contextlib import contextmanager


label_map = {0 : 'flag'}
final_pred_map = {-1:-1, 0:'absent', 1:'weak', 2:'strong'}
allowed_extensions = ['png','PNG','jpg','jpeg']


join_path = lambda *l: os.sep.join(l)

@contextmanager
def cwd(*l):
    path = os.sep.join(l)
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
        
        

def bulk_pred_2step(folder_path, th=0.5):

  #grabbing pictures only
  imgs = []
  for ext in allowed_extensions:
    imgs.extend(glob.glob(join_path(folder_path, '*.'+ext)))

  #sanity check
  if len(imgs) != len(os.listdir(folder_path)):
    print('WARNING: some files where excluded from input dir because where not recognized as images extensions.')

  result = {}
  for img in imgs:
    try:
      img_name = os.path.basename(img)
      img_class = img_pred_2step(img, th)
      result[img_name] = img_class
    except Exception as e:
      print('ERROR: failer to process image', img_name)
  return result


def img_pred_2step(img_path, th=0.5, v=True, debug=True):
    '''
    Predicts wind intensity from an image, according to the scale no wind, weak wind and strong wind (with respective class absent, weak, strong). Seeks flags, and based on the flags predicts the intensity. Returns -1 if no flags are found.
    INPUT:
        img_path: path to image
        th: acceptance threshold for flag detection algorithm
        v: verbose
    OUTPUT:
        Predicted wind intensity for the whole image. -1 if no flag is found.
    '''
    # run detector
    img_name, ext = os.path.splitext(os.path.basename(img_path))

    retina_model = models.load_model(join_path('models','retinanet_model.h5'), backbone_name='resnet50')
    image = read_image_bgr(img_path)
    res = predict_retinanet(retina_model, image, th)

    # if no flag is found
    if res.shape[0] == 0:
        return -1

    #output predicted image BBox, if desired
    if v or debug:
        ann_img = annotate_image(image, label_map, res)
        cv2.imwrite(join_path('out','annot_' + img_name + ext), ann_img)
    
    if debug:
        plt.axis("off")
        plt.imshow(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
        plt.show()
        print(res)

    #extract flag
    flags, _ = get_flags(img_path, res, new_dim=(240,240))

    #run classifier
    effnet_model = load_model(join_path('models','efficientnet_final.h5'))

    out_class = np.zeros((len(flags), 3))
    for i, flag in enumerate(flags):
        preds = effnet_model.predict(np.expand_dims(flag, axis=0))
        out_class[i,:] = preds

        if debug:
            plt.figure()
            print(preds)
            plt.axis("off")
            plt.imshow(flag.astype(int))
            plt.show()
            print((out_class*100).astype(int))

    # combine output
    avg = np.mean(out_class, axis=0)
    final_pred = np.argmax(avg)

    if debug:
        print((avg*100).astype(int))
        print(final_pred)
    return final_pred_map[final_pred]

############ RETINANET #####################################
def predict_retinanet(net,image, th=0.5):
        '''
        INPUT
                net: trained Retinanet model 
                imgage: image in BGR format

        OUTPUT
                Returns:
                - the bounding boxes as a np.array. Each row is a bounding box, 
                each column is (x_min, y_min, x_max, y_max)
                scores: confidence of each box
                labels: labels associated to each box
        '''
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = net.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        mask = scores[0] > th
        boxes = boxes[0][mask]
        scores = scores[0][mask]
        labels = labels[0][mask]
        return np.hstack([boxes, scores[:,np.newaxis], labels[:,np.newaxis]])

        
def annotate_image(image, label_map, boxes):
        '''
        INPUT
                imgage: image in BGR format
                label_map: dictionary with possible labels as keys and integers as values
                boxes: the bounding boxes as a np.array. Each row is a bounding box, 
                             each column is (x_min, y_min, x_max, y_max)
                scores: confidence of each box
                labels: labels associated to each box
        OUTPUT
                Returns the image with the bounding box
        '''
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        for box in boxes:                
                b = box.astype(int)
                caption = "{} {:.3f}".format(label_map[box[5]], box[4])
                cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255,0,0), 2, cv2.LINE_AA)
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                
        return draw
        

def get_flags(img_path,boxes,ratio=1.1,xml=True, new_dim=None):
    '''
    extracts only the detected flags from an image, after enlarging the bounding boxes.
    INPUT
            img_path =
            boxes = numpy array of bounding boxes, as
                    if xml=True: (xmin, ymin, xmax, ymax, ...)
                    if xml=False: (x, y, w/2, h/2, ...)
            ratio = ratio by which the boxes are enlarged
            xml = True if boxes from the annotated xml files, False if boxes from object detection
    OUTPUT
            flags = list of cropped images, as numpy arrays
            labels = if xml = True, list flag labels from manual annotation; if xml = False, an empty list.
    '''

    img = load_img(img_path)
    boxes = enlarge_boxes(boxes,ratio,xml)
    boxes = convert_c_bbox_to_corners(boxes)
    
    flags = []
    labels = []

    for box in boxes:
        im = img.crop(box[:4])
        if new_dim is not None:
                im = im.resize(new_dim)
        flags.append(img_to_array(im))
        if xml:
            labels.append(int(box[4]))

    return flags,labels
    

def enlarge_boxes(boxes,ratio=1.1,xml=True):
    '''
    enlarges bounding boxes by specified ratio.
    INPUT
            boxes = numpy array of bounding boxes, as
                    if xml=True: (xmin, ymin, xmax, ymax, ...)
                    if xml=False: (x, y, w/2, h/2, ...)
            ratio = ratio by which the boxes are enlarged
            xml = True if boxes from the annotated xml files, False if boxes from object detection
    OUTPUT
            numpy array of enlarged bounding boxes, as (x, y, w'/2, h'/2, ...)
    '''
    if xml:
        boxes = convert_corners_to_c_bbox(boxes)

    boxes[:,2] *= ratio
    boxes[:,3] *= ratio

    return boxes
    
    
def convert_corners_to_c_bbox(boxes):
        '''
        INPUT
                numpy array of bounding boxes, as (xmin, ymin, xmax, ymax, ...)
        OUPUT
                numpy array of bounding boxes, as (x, y, w/2, h/2, ...)
        '''
        x = (boxes[:,0] + boxes[:,2]) / 2
        y = (boxes[:,1] + boxes[:,3]) / 2
        w_2 = (boxes[:,2] - boxes[:,0]) / 2
        h_2 = (boxes[:,3] - boxes[:,1]) / 2
        return np.hstack([x[:,np.newaxis], y[:,np.newaxis], w_2[:,np.newaxis], h_2[:,np.newaxis], boxes[:,4:]])
        
        
def convert_c_bbox_to_corners(boxes):
        '''
        INPUT
                numpy array of bounding boxes, as (x, y, w/2, h/2, ...)
        OUPUT
                numpy array of bounding boxes, as (xmin, ymin, xmax, ymax, ...)
        '''
        xmin = boxes[:,0] - boxes[:,2]
        xmax = boxes[:,0] + boxes[:,2]
        ymin = boxes[:,1] - boxes[:,3]
        ymax = boxes[:,1] + boxes[:,3]
        return np.hstack([xmin[:,np.newaxis], ymin[:,np.newaxis], xmax[:,np.newaxis], ymax[:,np.newaxis], boxes[:,4:]])