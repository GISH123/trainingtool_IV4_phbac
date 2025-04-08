#2023.05.05  以TF2 讀取TF2的savedmodel:pass , 以TF2修改輸出op_name:fail  , identity() 禁止存取問題 scope error ,
import os ,shutil
import numpy as np
import tensorflow as tf
from models.inception_modules import Stem, InceptionBlockA, InceptionBlockB, InceptionBlockC, ReductionA, ReductionB
from configuration import NUM_CLASSES
from prepare_data import load_and_preprocess_image
# saved_model_cli  show --dir  ./DIYmodel   --all


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.stem = Stem()
        self.inception_a = self.build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = self.build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = self.build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax   )
        
        
        
    def call(self, inputs, training=None):
        #a, b = inputs   
        #hidden = self.dense_1(a + b)
        #hidden = self.dropout(hidden, training=training)
        #logits = self.dense_2(hidden)

        #x0= inputs
        x = self.stem(inputs, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        y = self.fc(x)
        
        #input_=tf.identity(x0, name= 'in_new')
        output_scorelist=tf.identity(y, name= 'score')
       
        return   output_scorelist

        #return logits 

    @tf.function(
        input_signature=[(
        
                                tf.TensorSpec([1, 299,299,3], name='image_tensor', dtype=tf.float32)
                                #tf.TensorSpec([1, 299,299,3], name='tensor_a', dtype=tf.float32),
                                #tf.TensorSpec([None, 10], name='b', dtype=tf.float32)
                                   
                        )]  ,
      )
    def sever(self, inputs):
        scorelist=self.call(inputs, training=None)
        #class_i   =tf.argmax(scorelist ,axis=1)
        
        #max_score =tf.reduce_max(output)
       
        #class_i = tf.identity(class_i, name='class' )
        #score = tf.identity(score, name='score' )
        
        #return { "class":class_i ,"score":  scorelist  }   # define  output_name
        return  { "score":  scorelist  }                    # define  output_name
        
        #return  output
    
    def build_inception_block_a(self,n):
        block = tf.keras.Sequential()
        for _ in range(n):
            block.add(InceptionBlockA())
        return block


    def build_inception_block_b(self,n):
        block = tf.keras.Sequential()
        for _ in range(n):
            block.add(InceptionBlockB())
        return block


    def build_inception_block_c(self,n):
        block = tf.keras.Sequential()
        for _ in range(n):
            block.add(InceptionBlockC())
        return block


    def __repr__(self):
        return "InceptionV4"

    @tf.function()
    def addNameToTensor(self, someTensor, theName):
        #with tf.control_dependencies( None ):
        out_new=tf.identity(someTensor, name=theName)
        return out_new






def  Predict_forzenPb_TF2 (modelfile_path ,input_tensordata_Mat ,input_op_name = 'image_tensor' ,  output_op_name='Identity'  ) :
    
    
    #---------Load forzen .pb--------------
    print('-------------------Load  forzen .pb-------------------------')
    # TF2 讀取 forzen  和 推論  也是沿用 TF1 session  API 
    with tf.compat.v1.gfile.FastGFile(modelfile_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
            # 將模型轉換為 TensorFlow 2 的函數

            #for op in graph.get_operations(): print("Operation Name :",op.name)             # Operation name
            #for node_i in graph_def.node:print( 'node Name=' ,node_i.name)
        
            
            print('----Check out the input placeholders:')
            nodes_input = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
            for node in nodes_input:print(node)
            #nodes_output = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Identity')]
            #for node in nodes_output:print(node)
            
            inputs_ = graph.get_tensor_by_name(input_op_name +':0')
            outputs_ = graph.get_tensor_by_name(output_op_name +':0')
            a=1

            #---------TF1--------------------
            # TF1 的 inputdata 不是tensor  , 使用時得加一個維度          
            with tf.compat.v1.Session(graph=graph) as sess:
                input_tensordata_Mat_4d = input_tensordata_Mat[np.newaxis,:]   #add 1 dim 
                output_pred = sess.run(outputs_, feed_dict={ inputs_: input_tensordata_Mat_4d   })
                print( 'forzenPb Predict :' , output_pred)
            a=1
            




#========================================================
# main
#========================================================
#isLoadsavemodel=False
#isLoadsavemodel=True     #(to load model_TF1)
isLoadsavemodel=False   #( create  model_TF2   convert to TF1)
saved_model_path=[]
#if   isLoadsavemodel==True :saved_model_path="./DIYmodel_TF1"
if   isLoadsavemodel==False : saved_model_path="./DIYmodel"


model = MyModel() 
#isLoadsavemodel=False
print('Create model successful ,',saved_model_path)
#if isLoadsavemodel :
#  model = tf.saved_model.load(saved_model_path)
#  print('Load model successful ,',saved_model_path)

    


#data1 = tf.ones((2, 10))
#data2 = tf.ones((2, 10))
#out = model((image_tensor, data2), training=True)


#-------model   predict--------------------------
test_image_dir='testone/c1.jpg'
import cv2
image_raw_mat = cv2.imread('./'+test_image_dir)
image_raw_mat=cv2.resize(image_raw_mat,(299,299))

image_raw = tf.io.read_file(filename='./'+test_image_dir)

image_tensor = load_and_preprocess_image(image_raw)
image_tensor = tf.expand_dims(image_tensor, axis=0)

output_  = model((image_tensor), training=True)
print('TF2 create model , pred_out=',output_)
print('----------------------')
#output_tensor_new=model.addNameToTensor(output_, 'output_new')  # could not access   : touched  but not work for name 


#max_index=tf.argmax(pred_out ,axis=1)
#max_value=tf.reduce_max(pred_out)
#print('max_index=',max_index   )
#print('max_value=',max_value   )

#model.save(saved_model_path )



isSavedmodel=True
#isSavedmodel=False
if isLoadsavemodel==True :     isSavedmodel=False
if isSavedmodel:
    #-----savedmodel_TF2------------
   
    model.save(saved_model_path, signatures={"serving_default": model.sever    })
    print('savedmodel TF2 saved: ' ,saved_model_path)



isForzen=True
#isForzen=False
if  isForzen:
    #------------------------------------------------------------
    # 加载已训练好的模型
    loaded_model = tf.keras.models.load_model(saved_model_path)
    session_model =loaded_model.signatures['serving_default']


    input_names = [t.name for t in   session_model.inputs ]
    output_names = [t.name for t in  session_model.outputs ]
    output_structured_tensor_name=  list ( session_model.structured_outputs.keys() ) 
    #structured_outputs= list (       session_model.structured_outputs.keys() ) 
    print('---Load saved_model ---------')
    print('input_names=',input_names[0])
    print('output_names=',output_names[0])
    print('structured_outputs_name=' ,output_structured_tensor_name[0])
    

    # Load savemodel to Graph 转换为图
    input_tensor_name =  session_model.inputs[0].name.replace(':0','')
    output_tensor_name =  session_model.outputs[0].name.replace(':0','')
    output_structured_tensor_name=  list ( session_model.structured_outputs.keys() )[0] 

    #----------修改輸出 name --------------- fail
    input_tensor =  session_model.graph.get_tensor_by_name(input_tensor_name+ ':0'    )   #image_tensor:0
    output_tensor = session_model.graph.get_tensor_by_name(output_tensor_name + ':0'  )   #Identity:0
#    output_structured_tensor = session_model.graph.get_tensor_by_name(output_structured_tensor_name+':0'   )   # not found  score:0
    
    #output_tensor_new=   tf.identity(    output_tensor   , name='output_tensor_new')   # could not access   : fail
    #output_tensor_new=model.addNameToTensor(output_tensor, 'output_new')  # could not access                : fail
    #input_tensor_new = tf.identity(input_tensor, name='input_tensor_new') # could not access               : fail
    
    # 將新的模型 signature 儲存為 savedmodel
    #modified_model = tf.keras.Model(inputs= input_tensor, outputs= {  'out_new' :output_tensor }  )    #fail
    #tf.saved_model.save(modified_model, 'DIYmodel_TF1')
    #print('./DIYmodel_TF1  , saved model :  output_tensor_name_new    ,  over~')





    #---------将具体函数转换为可执行图函数  Convert the concrete function to a TensorFlow graph---------
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
    #---------convert to forzen_graph-----------
    input_op_name=input_tensor_name  #'image_tensor'
    #output_op_name='score'
    output_op_name=output_tensor_name
    #output_op_name=output_structured_tensor_name
    
    inceptinoV4_spec_input =tf.TensorSpec(shape=[1,299, 299, 3], dtype=tf.float32, name=input_op_name ) #'image_tensor'
    #inceptinoV4_spec_output =tf.TensorSpec(shape=[1,14], dtype=tf.float32, name=output_op_name )        #'score'
    
    
    session_full_model = tf.function(session_model).get_concrete_function(inceptinoV4_spec_input  )  # 相當於TF1  with session()
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(session_full_model)
    
    print('-----------frozen_func  .inputs   .outputs '  )
    for intput in  frozen_func.inputs : print('intput_tensors_name=' ,intput.name)                      #image_tensor:0
    for output in  frozen_func.outputs : print('output_tensors_name=' ,output.name)                     #Identity:0
    for output in  frozen_func.structured_outputs : print('structured_output_tensors=' ,output.name)    #Identity:0
       
    #---------save forzen  .pb --------------
    frozen_out_path='frozen_models'
    frozen_graph_filename='frozen_model.pb'
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=frozen_graph_filename,
                    as_text=False
                )
    print('------save  forzen .pb-------   over~')

    #---------Predict_forzenPb_TF2----------------
    forzen_modelfile_path= os.path.join(frozen_out_path , frozen_graph_filename )
    Predict_forzenPb_TF2 (forzen_modelfile_path ,image_raw_mat ,input_op_name ,output_op_name )  # forzenPb  仍需 TF1 的session_run()  沒有 TF2的API




    



    
      
    


