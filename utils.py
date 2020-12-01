import numpy as np
import cv2
import os
import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from libs.retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from libs.retinanet.keras_retinanet import models
import matplotlib.pyplot as plt
import logging
from PIL import Image

logger = logging.getLogger('windspeed')
label_map = {0 : 'flag'}
final_pred_map = {-1:-1, 0:'absent', 1:'weak', 2:'strong'}
allowed_extensions = ['.png','.PNG','.jpg','.jpeg']
join_path = lambda *l: os.sep.join(l)
                

def bulk_pred_2step(folder_path, th=0.5, steps=False, out_dir='out'):
    '''
    Predicts wind intensity for all images in folder, according to the scale no wind, weak wind and strong wind (with respective class absent, weak, strong). Seeks flags, and based on the flags predicts the intensity. Returns -1 if no flags are found.
    INPUT:
        img_path: path to image
        th: acceptance threshold for flag detection algorithm
        steps: store intermediate steps to disk (input image with predicted flag confidence and input image with predicted wind intesity, per flag)
    OUTPUT:
        Dictionary, containing mapping image name -> predicted wind intensity
    '''
    #grabbing pictures only
    imgs = []
    for ext in allowed_extensions:
        imgs.extend(glob.glob(join_path(folder_path, '*'+ext)))

    #sanity check
    if len(imgs) != len(os.listdir(folder_path)):
        logger.warning('some files where excluded from input dir because where not recognized as images extensions: ' + ', '.join(allowed_extensions))
        
    #load models only once
    retnet = models.load_model(join_path('models','retinanet_model.h5'), backbone_name='resnet50')
    
    effnet = load_model(join_path('models','efficientnet_final.h5'))

    result = {}
    for img in imgs:
        try:
            img_class = _pred_2step_img(retnet, effnet, img, th, steps, out_dir)
            result.update(img_class)
        except Exception as e:
            logger.error('failer to process image', img)
    return result


def img_pred_2step(img_path, th=0.5, steps=False, out_dir='out'):
    '''
    Predicts wind intensity from an image, according to the scale no wind, weak wind and strong wind (with respective class absent, weak, strong). Seeks flags, and based on the flags predicts the intensity. Returns -1 if no flags are found.
    INPUT:
        img_path: path to image
        th: acceptance threshold for flag detection algorithm
        steps: store intermediate steps to disk (input image with predicted flag confidence and input image with predicted wind intesity, per flag)
    OUTPUT:
        Dictionary, containing mapping image name -> predicted wind intensity
    '''
    # run detector
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    
    if ext not in allowed_extensions:
        logger.warning('the path supplied "{}" is not a recognized image.'.format(img_name+ext))
        return {}

    retnet = models.load_model(join_path('models','retinanet_model.h5'), backbone_name='resnet50')
    
    effnet = load_model(join_path('models','efficientnet_final.h5'))
            
    return _pred_2step_img(retnet, effnet, img_path, th, steps, out_dir)
    
    
def _pred_2step_img(retnet, effnet, img_path, th=0.5, steps=False, out_dir='out'):

    img_name = os.path.basename(img_path)
    image = read_image_bgr(img_path)
    res = predict_retinanet(retnet, image, th)

    # if no flag is found
    if res.shape[0] == 0:
        return {img_name:-1}

    #output predicted image BBox, if desired
    if steps:
        ann_img = annotate_image(image, label_map, res)
        cv2.imwrite(join_path(out_dir,'flag_' + img_name), ann_img)

    #extract flag
    flags, _ = get_flags(img_path, res, new_dim=(240,240))

    #run classifier
    out_class = np.zeros((len(flags), 3))
    for i, flag in enumerate(flags):
        preds = effnet.predict(np.expand_dims(flag, axis=0))
        out_class[i,:] = preds
    
    #output predicted wind intensity, by flag
    if steps:
        res[:,4] = np.max(out_class, axis=1)
        res[:,5] = np.argmax(out_class, axis=1)
        ann_img = annotate_image(image, final_pred_map, res)
        cv2.imwrite(join_path(out_dir,'wind_' + img_name), ann_img)

    # combine output
    avg = np.mean(out_class, axis=0)
    final_pred = final_pred_map[np.argmax(avg)]
            
    return {img_name:final_pred}


def video_pred_2step(video_path, out_dir='out', th=0.5, codec='mp4v', ext='.mp4'):
    try:
        vin = cv2.VideoCapture(video_path)
    except Exception as e:
        logger.error('Failed to open video "{}". {}'.format(video_path, e))
        return 
        
    if not vin.isOpened():
        logger.error('Failed to open video "{}"'.format(video_path))
        vin.release()
        return
    
    video_name,_ = os.path.splitext(os.path.basename(video_path))
    
    try:
        fps = vin.get(cv2.CAP_PROP_FPS)
        width = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res=(int(width), int(height))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vout = cv2.VideoWriter(join_path(out_dir,'wind_' + video_name + ext), fourcc, fps, res)
        _video_pred_2step(vin, vout, th)
    except Exception as e:
        logger.error('Failed to open video "{}". {}'.format(video_path, e))
    finally:
        vin.release()
        vout.release()
    
    
def _video_pred_2step(vin, vout, th):
    retnet = models.load_model(join_path('models','retinanet_model.h5'), backbone_name='resnet50')
    
    effnet = load_model(join_path('models','efficientnet_final.h5'))
    
    while True:
        try:
            is_success, frame = vin.read()
        except cv2.error:
            continue
        if not is_success:
            break
        
        res = predict_retinanet(retnet, frame, th)

        # if no flag is found write frame
        if res.shape[0] == 0:
            vout.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        #extract flag
        frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        flags, _ = _get_flags(frame_PIL, res, ratio=1.1, xml=True, new_dim=(240,240))
        

        #run classifier
        out_class = np.zeros((len(flags), 3))
        for i, flag in enumerate(flags):
            preds = effnet.predict(np.expand_dims(flag, axis=0))
            out_class[i,:] = preds
        
        res[:,4] = np.max(out_class, axis=1)
        res[:,5] = np.argmax(out_class, axis=1)
        ann_img = annotate_image(frame, final_pred_map, res)
        
        image = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
        vout.write(image)



def predict_retinanet(net,image, th=0.5):
    '''
    INPUT
        net: trained Retinanet model 
        image: image in BGR format

    OUTPUT
        Returns the bounding boxes as a np.array. Each row is a bounding box, each column is
        (x_min, y_min, x_max, y_max, score, label)
        label: numerical id of the class
        score: confidence of the prediction
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
        boxes: the bounding boxes as a np.array. Each row is a bounding box, each column is (x_min, y_min, x_max, y_max, score, label)
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
    return _get_flags(img, boxes, ratio, xml, new_dim)
    

def _get_flags(img, boxes, ratio, xml, new_dim):
    #img: img in PIL format
    
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
                