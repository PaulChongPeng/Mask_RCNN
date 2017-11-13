import os
import skimage.io
import time
import coco
import model as modellib
import tensorflow as tf
import numpy as np
import visualize

flags = tf.app.flags
flags.DEFINE_string('IMAGE_DIR',
                    '/Users/paul/Data/dish/coco',
                    'weight file path.')
flags.DEFINE_integer('BATCH_SIZE', '2', 'batch size')
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
                visualize.save_image(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'], r['scores'],
                           coco_class_names, filter_classs_names=food_class_names,
                           scores_thresh=prob_thresh, mode=4)
            image_batch = []
            image_name_batch = []

        image_path = os.path.join(IMAGE_DIR, image_name_list[i])
        image = skimage.io.imread(image_path)
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
            visualize.save_image(image_batch[j], image_name_batch[j], r['rois'], r['masks'], r['class_ids'], r['scores'],
                       coco_class_names, filter_classs_names=food_class_names,
                       scores_thresh=prob_thresh, mode=0)

        image_batch = []
        image_name_batch = []

    print("cost time %f" % (time.time() - time_start))
    result_file.close()


if __name__ == '__main__':
    tf.app.run()
