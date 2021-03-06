# paths
qa_path = 'data'  # directory containing the question and annotation jsons
train_path = '../data/mscoco/train2014'  # directory of training images
val_path = '../data/mscoco/val2014'  # directory of validation images
test_path = '../data/mscoco/test2015'  # directory of test images
preprocessed_path = '/10707data/resnet-7x7.h5'  # path where preprocessed features are saved to and loaded from
numpy_path = './data/features_id.npz'
vocabulary_path = 'data/vocab/vocab.json'  # path where the used vocabularies for question and answers are saved to


task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 224 #224 for resnet 2048*7*7 filter adoption #448 for google paper  # scale shorter end of image to this size and centre crop
output_size = 7  # size of the feature maps after processing through a network
output_features = 2048 # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
word_embed_dim = 300
image_embed_dim = 2048

# training config
epochs = 100
batch_size = 100
initial_lr = 0.001 # default Adam lr
initial_embed_lr = 0.001
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
cuda = True
val_interval = 10
decay_step = 5
decay_size = 0.1
rn_conv_channel = 50
