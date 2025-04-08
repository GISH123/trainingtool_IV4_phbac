# cmd
#   tensorboard --logdir=board

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == '__main__':

    modelfile_path = os.path.join("frozen_models","frozen_model.pb") #请将这里的pb文件路径改为自己的
    graph=tf.Graph()
    with graph.as_default():
        graph = tf.get_default_graph()
        

    with tf.Session(graph=graph ) as sess:
        #---------predict .pb-----------------
        with tf.gfile.FastGFile(modelfile_path,'rb') as f:
        #with open( modelfile_path, 'rb') as f:
            sess.graph.as_default()
            
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='graph')
            summaryWriter = tf.summary.FileWriter('log/', graph)


    #os.system(' tensorboard --logdir log ' )
    #   tensorboard --logdir log