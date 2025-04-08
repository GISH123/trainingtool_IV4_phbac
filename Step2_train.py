from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras as nn
import math
import os
import argparse
import logging
import numpy as np

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, save_model_dir, save_every_n_epoch
from prepare_data import generate_datasets, load_and_preprocess_image
from models import get_model


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logging level to INFO

# Create file handler and set formatter
file_handler = logging.FileHandler('train.log')  # Specify the file name
file_handler.setLevel(logging.INFO)  # Set logging level to INFO for the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger
logger.addHandler(file_handler)


def forzenGraph(model ):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(  tf.TensorSpec(full_model.inputs[0].shape, full_model.inputs[0].dtype)    )

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]

    print("Frozen model layers: ")
    for layer in layers: print(layer)


    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./frozen_models",
                    name="_frozen_graph.pb",
                    as_text=False)


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()
    image_paths = features['image_path'].numpy()

    return images, labels, image_paths


parser = argparse.ArgumentParser()
parser.add_argument("--idx", default=0, type=int)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    args = parser.parse_args()

    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    logger.info(f'train_dataset : {train_dataset}')
    logger.info(f'valid_dataset : {valid_dataset}')

    # create model
#   import show_model_list
#   show_model_list.show_model_list()
    args.idx=14     #InceptionV4: 14
    model = get_model(args.idx)
    #print_model_summary(network=model)

    #----load pretrain model--------
    if os.path.exists(save_model_dir)==False: os.mkdir(save_model_dir)
        
    if not os.listdir(save_model_dir): 
        print("Directory is empty. no Load model...",save_model_dir)
    else:
        model = tf.saved_model.load(save_model_dir) 
        print("Loaded savedmodel ...", save_model_dir)


    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = nn.optimizers.Adam(learning_rate=1e-3)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    valid_acy_str='0'

    for epoch in range(EPOCHS):
        train_step_count = 0
        for features in train_dataset:
            train_step_count += 1
            images, labels, paths = process_features(features, data_augmentation=True)
            #images, labels = process_features(features, data_augmentation=False)
            # logger.info(f'labels : {labels} , paths : {paths}')
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, train accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     train_step_count,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))

        valid_step_count = 0
        last_valid_accuracy = 0
        for features in valid_dataset:
            valid_step_count += 1
            valid_images, valid_labels, valid_paths = process_features(features, data_augmentation=False)
            # logger.info(f'valid_labels : {valid_labels} , valid_paths : {valid_paths}')
            valid_step(valid_images, valid_labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     valid_step_count,
                                                                                     math.ceil(valid_count / BATCH_SIZE),
                                                                                     valid_loss.result().numpy(),
                                                                                     valid_accuracy.result().numpy()))
            # 2024/05/10 check if valid_accuracy_drops, and record the data point that results worse accuracy
            accuracy_diff = last_valid_accuracy - valid_accuracy.result().numpy()
            if accuracy_diff > 0 :
                logger.info(f'current step results in worse valid_accuracy, accuracy diff is {accuracy_diff}, check epoch : {epoch}, step : {valid_step_count}, image_path : {valid_paths}')
            # update last_valid_accuracy
            last_valid_accuracy = valid_accuracy.result().numpy()


        valid_accuracy_percentage= valid_accuracy.result().numpy()
        valid_acy_str= str(valid_accuracy_percentage)
        #print( int(valid_accuracy_percentage*1000)/1000 )
        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        #if epoch % save_every_n_epoch == 0:
        #    model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')
        #--------early stop-----------------------JK,2023.04.18
        
        
        if  valid_accuracy_percentage >0.9987 : 
        #if  valid_accuracy_percentage >0.9977 : 
        #if  valid_accuracy_percentage >0.9987 : 
        #if  valid_accuracy_percentage >0.4 : 
            print('valid_accuracy_percentage=',valid_accuracy_percentage)
            #model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf') 
            tf.saved_model.save(model, save_model_dir+"_validAcc_"+valid_acy_str)            
                       
            #forzenGraph(model._export_to_saved_model_graph() ,valid_acy_str)
            print('[model saved] :',save_model_dir+"_validAcc_"+valid_acy_str )
            if valid_accuracy_percentage >0.9999 : break
        else : 
            print('[model not saved] :',save_model_dir+"_validAcc_"+valid_acy_str )   

         
            
    #---final round to save -----------------------------        
    # save weights
    #model.save_weights(filepath=save_model_dir+"model_"+valid_acy_str, save_format='tf')
    #save the whole final model
    #model.save("./DIYmodel", signatures={"serving_default": model.sever})
    tf.saved_model.save(model, save_model_dir+"_final_"+valid_acy_str)
    print('[final round  model saved] :',save_model_dir+"_final_" ,valid_acy_str )
    

    
    #---convert to tensorflow lite format
    #model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)



    '''
    #---run predict by reload savedmodel---------------------------------
    test_image_dir='testone/one.png'
    image_raw = tf.io.read_file(filename='./'+test_image_dir)
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    pred_out = model((image_tensor), training=True)
    print(pred_out)

    '''