import os

import tensorflow as tf
from prepare_data import load_and_preprocess_image



class CustomModuleWithOutputName(tf.Module):
  def __init__(self):
    super(CustomModuleWithOutputName, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)]   )
  def __call__(self, x):
    return {'custom_output_name': x * self.v}


if __name__ == '__main__':
    saved_model_path="./DIYmodel"
    module_output = CustomModuleWithOutputName()
    call_output = module_output.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
 
    tf.saved_model.save(module_output, saved_model_path,signatures={'serving_default': call_output})
    model = tf.keras.models.load_model(saved_model_path)
 
    
    
    output=model.signatures['serving_default'].outputs   
    structured_outputs=model.signatures['serving_default'].structured_outputs   
    
    print("---------------Load saved model  output : ")
    print('output_name=' ,output)
    print('structured_outputs_name=' ,structured_outputs)


    # 转换为图结构
    graph_func = model.signatures['serving_default']


    # Convert the concrete function to a TensorFlow graph
    # 将具体函数转换为可执行图函数
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

    #spec_input =tf.TensorSpec(shape=[1,299, 299, 3], dtype=tf.float32, name='image_tensor' )
    spec_input=tf.TensorSpec([], tf.float32)
    full_model = tf.function(graph_func).get_concrete_function( spec_input )


    #---------convert to forzen_graph-----------
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    
    output_tensors = frozen_func.outputs
    print('output_tensors=' ,output_tensors)
    for output in  output_tensors :
        print('output_tensors_name=' ,output.name)
    

    print("-----------------Frozen model inputs: ")
    print(frozen_func.inputs )
    print("-----------------Frozen model outputs: ")
    print(frozen_func.outputs )

    
    frozen_out_path='./frozen_models'
    frozen_graph_filename='frozen_model'
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False
                  )