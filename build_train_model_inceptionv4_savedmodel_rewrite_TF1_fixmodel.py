#2023.05.02 ,JK 建立 自訂class模型後 以TF1的simple_save() 寫出模型檔方式 ，才可以指定內輸出層:name.
#2023.05.04  以TF1 讀取TF2的savedmodel:pass , 以TF1修改輸出op_name:pass , 但以TF1_sess轉存simple_save()後 重新載模型進行推論會缺少模型層. 
#2023.05.05  以TF1 讀取TF2的savedmodel:pass , 但以TF1_sess轉存 forzen.pb後 重新載模型進行推論:pass . 但不能修改輸出name.

import os ,shutil
import numpy as np
from models.inception_modules import Stem, InceptionBlockA, InceptionBlockB, InceptionBlockC, ReductionA, ReductionB
from configuration import NUM_CLASSES
from prepare_data import load_and_preprocess_image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# 定義 MyModel
class MyModel(tf.keras.Model):
    #def __init__(self):
    def __init__(self, **kwargs):    
        #--------------------
        super(MyModel, self).__init__(**kwargs)
        #self.dense1 = tf.keras.layers.Dense(units=64, activation='relu' )
        #self.dense2 = tf.keras.layers.Dense(units=10  )
        
        def build_inception_block_a(n):
            block = tf.keras.Sequential()
            for _ in range(n):
                block.add(InceptionBlockA())
            return block
        def build_inception_block_b(n):
            block = tf.keras.Sequential()
            for _ in range(n):
                block.add(InceptionBlockB())
            return block
        def build_inception_block_c(n):
            block = tf.keras.Sequential()
            for _ in range(n):
                block.add(InceptionBlockC())
            return block
        self.stem = Stem()
        self.inception_a = build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax   )

    #def call(self, inputs):
    def call(self, inputs, training=True):    

        #x0 = self.dense1(inputs)
        #y= self.dense2(x0)
        #x0= inputs
        x = self.stem(inputs, training=training )
        x = self.inception_a(x, training=training  )
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        y = self.fc(x )
                
        #input_=tf.identity(x0, name= 'image_tensor')
        output_=tf.identity(y, name= 'score')
        #input_=tf.identity(inputs, name= 'serving_default_image_tensor')
        return   output_ 

def Reload_Predict_TF1(loadmodel_path ,input_tensordata_Mat ):
    print('Reload Model path=',  loadmodel_path )
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    graph2=tf.Graph()
    with graph2.as_default():
        model = MyModel()       
        output_tenasor_name='StatefulPartitionedCall'
        input_tenasor_name='serving_default_image_tensor'
        #input_tenasor_name='serving_default_input_1'
              
        #-----------------model  save as TF1 
        #input_tensor = tf.keras.Input(shape=(64,))
        input_tensor = tf.keras.Input(shape=(299,299,3))
        output_ =model(input_tensor  )
        #print( 'TF1_pred_output=',output_)    
        #model.summary()
        with tf.Session(graph=graph2 ) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            meta_graph_def=tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING]  , loadmodel_path)
            
            #for op in sess.graph.get_operations()  : print(op.name)
            print('[model_path]=',  loadmodel_path)
            
            input_tensor_ori = sess.graph.get_tensor_by_name(input_tenasor_name+':0')
            output_tensor_ori = sess.graph.get_tensor_by_name(output_tenasor_name+':0') 
            session_pred = sess.run(output_tensor_ori, feed_dict={input_tensor_ori: [input_tensordata_Mat]})
            #session_pred = sess.run(output_, feed_dict={input_: [input_tensordata_Mat]})
            
            print('Reload model, TF1_session_pred=',session_pred)


def  Predict_ForzenPb_TF1( modelfile_path ,input_tensordata_Mat  ,input_tensor_name='serving_default_image_tensor' ,output_tensor_name='StatefulPartitionedCall' ):
    
    with tf.Session(graph=tf.Graph() ) as sess:
        #---------predict .pb-----------------
        with tf.gfile.FastGFile(modelfile_path,'rb') as f:
        #with open( modelfile_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            #-----list op_names--------------
            #for op in graph.get_operations(): print("Operation Name :",op.name)             # Operation name
            #for node_i in graph_def.node:print( 'node Name=' ,node_i.name)
            #------predict--------------
            
            #output = tf.import_graph_def(graph_def, input_map={ 'a:0': inputdata }, return_elements=['out_new:0'])
            input_tensordata_Mat_4d = input_tensordata_Mat[np.newaxis,:]   #add 1 dim  
    
            input_tensor_name_0= input_tensor_name+':0'
            output_tensor_name_0 = output_tensor_name +':0'
            output = tf.import_graph_def(graph_def, input_map={ input_tensor_name_0 : input_tensordata_Mat_4d }, return_elements=[output_tensor_name_0])
            print( 'isPred_forzen : issave_forzen_pb  session_pred=' , sess.run(output))        # predict


def  model_save_TF1 (load_model_path_fromTF2 ,export_path ,input_tensordata_Mat) :
    graph=tf.Graph()
    with graph.as_default():
        
        model = MyModel()
        output_tenasor_name='StatefulPartitionedCall'
        input_tenasor_name='serving_default_image_tensor'
              
        #-----------------model  save as TF1 
        input_tensor = tf.keras.Input(shape=(299,299,3) ,dtype=tf.float32 )
        output_  =model(input_tensor  )  # must
    #    print( 'TF1_pred_output=',output_)
    
        #output_tensor_new = tf.identity(output_,'score')
        model.summary()

        with tf.Session(graph=graph ) as sess:
            
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            meta_graph_def=tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING]  , load_model_path_fromTF2)
            print('[model_path]=',  load_model_path_fromTF2)
            
            
            #for epoch in range(epochs):
            #    a=1
            #    #traing...
            #for op in sess.graph.get_operations(): print(op.name)
            input_tensor_ori = sess.graph.get_tensor_by_name(input_tenasor_name+':0')
            output_tensor_ori = sess.graph.get_tensor_by_name(output_tenasor_name+':0') 
            session_pred = sess.run(output_tensor_ori, feed_dict={input_tensor_ori: [input_tensordata_Mat]})
        
        
            print('Ori TF1_session_pred=',session_pred)
            

            #-----savemodel---------------
            #input_tenasor_name='serving_default_input_1'
            #input_tensor_new=tf.identity(input_tensor_ori, name=input_tenasor_name )   # modify the name is successful ,but actually not.

            #inputs_info = {input_tenasor_name: input_tensor_ori }
            #outputs_info = {output_tenasor_name: output_tensor_ori }
            
            inputs_info = {input_tenasor_name: input_tensor_ori }
            outputs_info = {output_tenasor_name: output_tensor_ori }
            signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs_info ,outputs=outputs_info)
            
                                                        
            #-----------mode save method1----------------------------
            if loadd_model_path.find('TF1')<0:  #when load TF2 savemodel  ,to save for TF1
                if os.path.exists(export_path) : shutil.rmtree(export_path)
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
        
                
                #save method1
                #tf.saved_model.save(model, export_path)   # savemodel reloaded  run :pass
                
                #save method2  
                tf.saved_model.simple_save(  sess , export_dir=export_path , inputs=inputs_info, outputs=outputs_info  )                  
                
                #save method3  TF1
                '''
                builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
                builder.add_meta_graph_and_variables(
                    sess=sess,
                    tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                    signature_def_map={               
                        tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
                    }
                    )
                builder.save()        
                '''
                
                print( 'TF1_model_svved ~')



                #------forzen save-------------------
                # forzen .Pb  是可以正常推論  但   存.pb沒法更改輸出層name    
                #issave_forzen_pb=True
                issave_forzen_pb=False
                if issave_forzen_pb:
                    if os.path.exists(export_path)==False :os.mkdir(export_path)
                    #----------save .pb  -----------------
                    save_forzen_model_name= 'saved_model_forzen.pb'
                    
                    #-----asign forzen 
                    from tensorflow.python.framework.graph_util import convert_variables_to_constants
                    frozen_graph = convert_variables_to_constants(sess, sess.graph_def, [output_tenasor_name])
                    
                    tf.train.write_graph(frozen_graph, export_path, save_forzen_model_name, as_text=False)  #.pb
                    print('Save_forzen_pb  .pb  successful~')


                    isPred_forzen =True
                    if  isPred_forzen :
                        #---------predict .pb-----------------
                        modelfile_path= os.path.join(export_path , save_forzen_model_name )
                        Predict_ForzenPb_TF1(modelfile_path ,input_tensordata_Mat, input_tenasor_name ,output_tensor_name=output_tenasor_name )
                        print('-------------Predict_ForzenPb_TF1()   over~----------------')
    return export_path





def  prepare_inputtensor() :
    #-------model   predict--------------------------
    test_image_dir='testone/c1.jpg'
    import cv2
    image_raw_mat = cv2.imread('./'+test_image_dir)
    image_raw_mat=cv2.resize(image_raw_mat,(299,299))
    
    image_raw = tf.io.read_file(filename='./'+test_image_dir)
    
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return  image_tensor

def  prepare_inputMat() :
    #-------model   predict--------------------------
    test_image_dir='testone/c1.jpg'
    import cv2
    image_raw_mat = cv2.imread('./'+test_image_dir)
    image_raw_mat=cv2.resize(image_raw_mat,(299,299))

    input_tensordata=np.array(image_raw_mat,dtype=np.float32)
    return  input_tensordata



if __name__ == '__main__':
    
    saved_model_path="DIYmodel"
    export_path =saved_model_path+'_TF1'

    #----------------model predict_TF2 
    #inputtensor_data=prepare_inputtensor() 
    #output_ ,input_ =model( inputtensor_data ,training=False )
    #print( 'TF2_pred_output=',output_)


    loadd_model_path=saved_model_path  #load TF2 model
    #loadd_model_path=export_path  #load TF1 model


    # Load from TF2_savedmodel   then to  TF1  Save the model
    input_tensordata_Mat=prepare_inputMat()
    savedmodelTF1_export_path=  model_save_TF1 ( loadd_model_path ,export_path ,input_tensordata_Mat ) 
    
    

    
    #----reload model and predict --------------------
    #model = tf.saved_model.load(export_path)
    #print('ReLoad model successful ,',export_path)
    #output_ ,input_ =model(input_tensor ,training=False )
    #print( 'ReLoad: TF1_pred_output=',output_)
    input_tensordata_Mat=prepare_inputMat()
    Reload_Predict_TF1(savedmodelTF1_export_path ,input_tensordata_Mat )   # fail



                




