import os ,shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np

'''
import tensorflow as tf
import numpy as np

def extract_tensors(signature_def, graph):
    output = dict()

    for key in signature_def:
        value = signature_def[key]

        if isinstance(value, tf.TensorInfo):
            output[key] = graph.get_tensor_by_name(value.name)

    return output

def extract_input_name(signature_def, graph):
    input_tensors = extract_tensors(signature_def['serving_default'].inputs, graph)
    #Assuming one input in model.
    key = list(input_tensors.keys())[0]
    return input_tensors.get(key).name

def extract_output_name(signature_def, graph):
    output_tensors = extract_tensors(signature_def['serving_default'].outputs, graph)
    #Assuming one output in model.
    key = list(output_tensors.keys())[0]
    return output_tensors.get(key).name

messages = [ "Some input text", "Another text" ]
new_text = np.array(messages, dtype=object)[:, np.newaxis]

model_dir = "./models/use/1"

with tf.Session(graph=tf.Graph()) as session:
    serve = tf.saved_model.load(session ,tags={'serve'}, export_dir=model_dir)

    input_tensor_name = extract_input_name(serve.signature_def, session.graph)
    output_tensor_name = extract_output_name(serve.signature_def, session.graph)

    result = session.run([output_tensor_name], feed_dict={input_tensor_name: new_text})
    print(result)
'''


class Adder(tf.Module):
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self,x):
        return  x+1
if __name__ == '__main__':
    
    saved_model_path=r'DIYmodel_TF1'
    save_model_name= 'saved_model.pb'
    #adder_out = Adder()
    

    a = tf.placeholder(dtype=tf.float32, shape=(1,1), name=r'a')
    b = tf.Variable(2, dtype=tf.float32, name=r'b')

    c = tf.add(a, b, name=r'c')
    d = tf.Variable(2, dtype=tf.float32, name=r'd')
    out_new = tf.multiply(c, d, name=r'out_new')


    isSaved_pb=True
    
    issaved_model_pb=True
    if isSaved_pb:
        #----------save .pb  -----------------
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if os.path.exists(saved_model_path): shutil.rmtree(saved_model_path)
            if os.path.exists(saved_model_path)==False :os.mkdir(saved_model_path)
            

            issaved_model_pb=True
            if issaved_model_pb:
                #----------------------------
                tf.saved_model.simple_save( session=sess, export_dir=saved_model_path, inputs={r'a':a}, outputs={r'out_new' : out_new} ,legacy_init_op=None	 )
                #tf.compat.v1.saved_model.save(adder_out,saved_model_path)   # same as TF2 model.save()
            
            issave_forzen_pb=False
            if issave_forzen_pb:
                #----------save .pb  -----------------
                #saver = tf.train.Saver()
                #saver.save(sess, saved_model_path)
        
                #-----asign forzen 
                frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['out_new'])
                tf.train.write_graph(frozen_graph, saved_model_path, save_model_name, as_text=False)  #.pb
            


    
    
    isPred =True
    if  isPred :
        with tf.Session(graph=tf.Graph()) as sess:
            modelfile_path= os.path.join(saved_model_path , save_model_name )
            if os.path.exists(modelfile_path):
                
                if issaved_model_pb:              
                    model=tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir=saved_model_path)
                    

                            
                    #-----list op_names--------------
                    for op in sess.graph.get_operations(): print(op.name)
                    #------predict--------------
                    input_tensor_name = sess.graph.get_tensor_by_name('a:0')
                    output_tensor_name = sess.graph.get_tensor_by_name('out_new:0')                
                    
                    #input_tensor_name = sess.graph.get_tensor_by_name('image_tensor:0')
                    #output_tensor_name = sess.graph.get_tensor_by_name('class:0')                
                    #output_tensor_name = sess.graph.get_tensor_by_name('score:0')                
                    
                    
                    
                    input_data = [[1.]]
                    session_pred = sess.run(output_tensor_name, feed_dict={input_tensor_name: input_data})
                    print( 'issaved_model_pb  session_pred=' ,   session_pred  )        # predict
                    
                
                if issave_forzen_pb:
                    with tf.gfile.FastGFile(modelfile_path,'rb') as f:
                    #with open( modelfile_path, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        sess.graph.as_default()

                        #-----list op_names--------------
                        for op in graph.get_operations(): print("Operation Name :",op.name)             # Operation name
                        #for node_i in graph_def.node:print( 'node Name=' ,node_i.name)
                        #------predict--------------
                        
                        #output = tf.import_graph_def(graph_def, input_map={ 'a:0': inputdata }, return_elements=['out_new:0'])        
                        output = tf.import_graph_def(graph_def, input_map={ 'a:0': 1. }, return_elements=['out_new:0'])
                        print( 'issave_forzen_pb  session_pred=' , sess.run(output))        # predict

                    
                    
        
        
