import os
import skimage.io
import time
import coco
import model as modellib
import tensorflow as tf
import numpy as np
import visualize
from PIL import Image, ImageFilter
import cv2

flags = tf.app.flags
flags.DEFINE_string('IMAGE_DIR',
                    '/Users/paul/Data/dish/coco',
                    'weight file path.')
flags.DEFINE_integer('BATCH_SIZE', '1', 'batch size')
flags.DEFINE_float('PROB_THRESH', '0.80', 'prob thresh')

FLAGS = flags.FLAGS


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = FLAGS.BATCH_SIZE


def get_centroid(id, boxes, masks, class_ids, class_names, filter_classs_names=None, scores=None, scores_thresh=0.1):
    N = boxes.shape[0]
    result = ""
    if not N:
        print("\n*** No instances to save *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        # Mask
        mask = masks[:, :, i]
        # print(mask)
        coor = np.where(mask == 1)
        area = len(coor[0])
        centroid = np.average(coor, 1)
        result = "%s%s, %d, %d, %d, %f, %s, %d, %f, %f\n" % (
            result, id, mask.shape[0], mask.shape[1], class_id, score, label, area, centroid[0], centroid[1])
    return result


def compute_mask_score(mask):
    image_area = mask.shape[0] * mask.shape[1]
    coor = np.where(mask == 1)
    area_score = len(coor[0]) / image_area
    centroid = np.average(coor, 1)
    centroid_score_x = 1 - (np.abs(centroid[0] - 0.5 * mask.shape[0]) / (0.5 * mask.shape[0]))
    centroid_score_y = 1 - (np.abs(centroid[1] - 0.5 * mask.shape[1]) / (0.5 * mask.shape[1]))
    centroid_score = (centroid_score_x + centroid_score_y) / 2
    mask_score = area_score * 0.6 + centroid_score * 0.4
    return mask_score


def background_blur(image, image_name, boxes, masks, class_ids, scores, class_names, filter_classs_names=None,
                    scores_thresh=0.1, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    best_mask = None
    best_mask_score = 0

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to blur *** \n" % (image_name))
        return
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        mask_score = compute_mask_score(masks[:, :, i])
        if mask_score > best_mask_score:
            best_mask = masks[:, :, i]
            best_mask_score = mask_score

    if best_mask is None:
        print("\n*** No instances in image %s to blur *** \n" % (image_name))
    else:
        '''
        image = image.astype(np.uint8)
        Image.fromarray(image).save(os.path.join(save_dir, '%s.jpg' % (image_name)), 'jpeg')
        blur_img = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=5))
        blur_img = np.array(blur_img)
        for c in range(3):
            blur_img[:, :, c] = np.where(best_mask == 1, image[:, :, c], blur_img[:, :, c])
        blur_img = Image.fromarray(blur_img)
        blur_img.save(os.path.join(save_dir, '%s_blur.jpg' % (image_name)), 'jpeg')
        '''

        '''
        kernel_size = (15, 15)
        sigma = 100
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % (image_name)),image)

        blur_img = cv2.GaussianBlur(image, kernel_size, sigma)
        best_mask = np.stack([best_mask,best_mask,best_mask],axis=2)
        #np.save('mask.npy',best_mask)
        #mask_img = Image.fromarray(best_mask).filter(ImageFilter.GaussianBlur(radius=5))
        #mask_img = np.array(mask_img)

        blur_img = image*best_mask + blur_img*(1-best_mask)
        cv2.imwrite(os.path.join(save_dir, '%s_blur.jpg' % (image_name)), blur_img)
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, '%s.jpg' % (image_name)), image)

        blur_img_0 = cv2.GaussianBlur(image, (7, 7), 0)
        blur_img_1 = cv2.GaussianBlur(blur_img_0, (9, 9), 0)
        blur_img_2 = cv2.GaussianBlur(blur_img_1, (11, 11), 0)
        blur_img_3 = cv2.GaussianBlur(blur_img_2, (13, 13), 0)
        blur_img_4 = cv2.GaussianBlur(blur_img_3, (15, 15), 0)

        best_mask = np.stack([best_mask, best_mask, best_mask], axis=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
        dilated_mask_0 = cv2.dilate(best_mask, kernel)
        dilated_mask_1 = cv2.dilate(dilated_mask_0, kernel)
        dilated_mask_2 = cv2.dilate(dilated_mask_1, kernel)
        dilated_mask_3 = cv2.dilate(dilated_mask_2, kernel)

        blur_img = image * best_mask + blur_img_0 * (dilated_mask_0 - best_mask) + blur_img_1 * (
            dilated_mask_1 - dilated_mask_0) + blur_img_2 * (dilated_mask_2 - dilated_mask_1) + blur_img_3 * (
            dilated_mask_3 - dilated_mask_2) + blur_img_4 * (1 - dilated_mask_3)
        cv2.imwrite(os.path.join(save_dir, '%s_blur.jpg' % (image_name)), blur_img)



def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    np.set_printoptions(threshold=np.inf)

    batch_size = FLAGS.BATCH_SIZE

    # Root directory of the project
    # ROOT_DIR = FLAGS.ROOT_DIR
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    # Download this file and place in the root of your
    # project (See README file for details)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Directory of images to run detection on
    # IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    IMAGE_DIR = FLAGS.IMAGE_DIR

    prob_thresh = FLAGS.PROB_THRESH

    image_name_list = os.listdir(IMAGE_DIR)

    result_file = open("result.txt", "w")

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                        'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                        'kite', 'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                        'teddy bear', 'hair drier', 'toothbrush']

    food_class_names = ['cup', 'bowl', 'sandwich', 'broccoli', 'hot dog', 'pizza', 'donut', 'cake']

    # Load a random image from the images folder
    time_start = time.time()

    '''
    image_path = '/Users/paul/Data/dish/coco/11.jpg'
    image = skimage.io.imread(image_path)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                coco_class_names, r['scores'])
    
    '''

    image_batch = []
    image_name_batch = []
    for i in range(len(image_name_list)):
        if len(image_batch) == batch_size:
            results = model.detect(image_batch, verbose=1)
            for j in range(batch_size):
                r = results[j]
                result_line = get_centroid(image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                           coco_class_names, filter_classs_names=food_class_names, scores=r['scores'],
                                           scores_thresh=prob_thresh)
                result_file.write(result_line)
                visualize.save_image(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                     r['scores'],
                                     coco_class_names, filter_classs_names=food_class_names,
                                     scores_thresh=prob_thresh, mode=0)
                background_blur(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                     r['scores'],
                                     coco_class_names, filter_classs_names=food_class_names,
                                     scores_thresh=prob_thresh)
            image_batch = []
            image_name_batch = []

        image_path = os.path.join(IMAGE_DIR, image_name_list[i])
        image = skimage.io.imread(image_path)
        #image = cv2.imread(image_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_batch.append(image)
        image_name_batch.append(image_name_list[i].split('.')[0])

    if len(image_batch) != 0:
        results = model.detect(image_batch, verbose=1)
        for j in range(len(image_batch)):
            r = results[j]
            result_line = get_centroid(image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                       coco_class_names, filter_classs_names=food_class_names, scores=r['scores'],
                                       scores_thresh=prob_thresh)
            result_file.write(result_line)
            visualize.save_image(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                 r['scores'],
                                 coco_class_names, filter_classs_names=food_class_names,
                                 scores_thresh=prob_thresh, mode=0)
            background_blur(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'],
                                 r['scores'],
                                 coco_class_names, filter_classs_names=food_class_names,
                                 scores_thresh=prob_thresh)

        image_batch = []
        image_name_batch = []

    print("cost time %f" % (time.time() - time_start))
    result_file.close()


if __name__ == '__main__':
    tf.app.run()
