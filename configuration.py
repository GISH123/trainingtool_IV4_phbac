
DEVICE = "gpu"   # cpu or gpu

# some training parameters
EPOCHS = 10
BATCH_SIZE = 8
NUM_CLASSES =  52+2 #52+1
#NUM_CLASSES = 53


#IMAGE_HEIGHT = 224
#IMAGE_WIDTH = 224

#------InceptionV4 :299x299x3-----------
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
CHANNELS = 3


save_model_dir = "saved_model"
save_every_n_epoch = 10
test_image_dir = "testone/one.jpg"

dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2


