import os ,shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np





def _tf_example_input_placeholder(input_shape=None):
  """Returns input that accepts a batch of strings with tf examples.
  Args:
    input_shape: the shape to resize the output decoded images to (optional).
  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_tf_example_placeholder = tf.keras.Input(tf.string, shape=[None], name='tf_example')

def _encoded_image_string_tensor_input_placeholder(input_shape=None):
  """Returns input that accepts a batch of PNG or JPEG strings.
  Args:
    input_shape: the shape to resize the output decoded images to (optional).
  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.tf.keras.Input(
      dtype=tf.string,
      shape=[None],
      name='encoded_image_string_tensor')

def _image_tensor_input_placeholder(input_shape=None):
  """Returns input placeholder and a 4-D uint8 image tensor."""
  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.keras.Input(dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor


#-------------------------------------------------
input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor': _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder
    }


def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
  """Writes SavedModel to disk.
  If checkpoint_path is not None bakes the weights into the graph thereby
  eliminating the need of checkpoint files during inference. If the model
  was trained with moving averages, setting use_moving_averages to true
  restores the moving averages, otherwise the original set of variables
  is restored.
  Args:
    saved_model_path: Path to write SavedModel.
    frozen_graph_def: tf.GraphDef holding frozen graph.
    inputs: A tensor dictionary containing the inputs to a DetectionModel.
    outputs: A tensor dictionary containing the outputs of a DetectionModel.
  """
  with tf.Graph().as_default():
    with tf.Session() as sess:

      tf.import_graph_def(frozen_graph_def, name='')

      builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

      tensor_info_inputs = {}
      if isinstance(inputs, dict):
        for k, v in inputs.items():
          tensor_info_inputs[k] = tf.saved_model.utils.build_tensor_info(v)
      else:
        tensor_info_inputs['inputs'] = tf.saved_model.utils.build_tensor_info(
            inputs)
      tensor_info_outputs = {}
      for k, v in outputs.items():
        tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

      detection_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs=tensor_info_inputs,
              outputs=tensor_info_outputs,
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      builder.add_meta_graph_and_variables(
          sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants
              .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  detection_signature,
          },
      )
      builder.save()





class DetectionResultFields(object):
  """Naming conventions for storing the output of the detector.
  Attributes:
    source_id: source of the original image.
    key: unique key corresponding to image.
    detection_boxes: coordinates of the detection boxes in the image.
    detection_scores: detection scores for the detection boxes in the image.
    detection_multiclass_scores: class score distribution (including background)
      for detection boxes in the image including background class.
    detection_classes: detection-level class labels.
    detection_masks: contains a segmentation mask for each detection box.
    detection_surface_coords: contains DensePose surface coordinates for each
      box.
    detection_boundaries: contains an object boundary for each detection box.
    detection_keypoints: contains detection keypoints for each detection box.
    detection_keypoint_scores: contains detection keypoint scores.
    detection_keypoint_depths: contains detection keypoint depths.
    num_detections: number of detections in the batch.
    raw_detection_boxes: contains decoded detection boxes without Non-Max
      suppression.
    raw_detection_scores: contains class score logits for raw detection boxes.
    detection_anchor_indices: The anchor indices of the detections after NMS.
    detection_features: contains extracted features for each detected box
      after NMS.
  """

  source_id = 'source_id'
  key = 'key'
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_multiclass_scores = 'detection_multiclass_scores'
  detection_features = 'detection_features'
  detection_classes = 'detection_classes'
  detection_masks = 'detection_masks'
  detection_surface_coords = 'detection_surface_coords'
  detection_boundaries = 'detection_boundaries'
  detection_keypoints = 'detection_keypoints'
  detection_keypoint_scores = 'detection_keypoint_scores'
  detection_keypoint_depths = 'detection_keypoint_depths'
  detection_embeddings = 'detection_embeddings'
  detection_offsets = 'detection_temporal_offsets'
  num_detections = 'num_detections'
  raw_detection_boxes = 'raw_detection_boxes'
  raw_detection_scores = 'raw_detection_scores'
  detection_anchor_indices = 'detection_anchor_indices'


def add_output_tensor_nodes(postprocessed_tensors,output_collection_name='inference_op'):
  """Adds output nodes for detection boxes and scores.
  Adds the following nodes for output tensors -
    * num_detections: float32 tensor of shape [batch_size].
    * detection_boxes: float32 tensor of shape [batch_size, num_boxes, 4]
      containing detected boxes.
    * detection_scores: float32 tensor of shape [batch_size, num_boxes]
      containing scores for the detected boxes.
    * detection_multiclass_scores: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_classes_with_background] for containing class
      score distribution for detected boxes including background if any.
    * detection_features: (Optional) float32 tensor of shape
      [batch, num_boxes, roi_height, roi_width, depth]
      containing classifier features
      for each detected box
    * detection_classes: float32 tensor of shape [batch_size, num_boxes]
      containing class predictions for the detected boxes.
    * detection_keypoints: (Optional) float32 tensor of shape
      [batch_size, num_boxes, num_keypoints, 2] containing keypoints for each
      detection box.
    * detection_masks: (Optional) float32 tensor of shape
      [batch_size, num_boxes, mask_height, mask_width] containing masks for each
      detection box.
  Args:
    postprocessed_tensors: a dictionary containing the following fields
      'detection_boxes': [batch, max_detections, 4]
      'detection_scores': [batch, max_detections]
      'detection_multiclass_scores': [batch, max_detections,
        num_classes_with_background]
      'detection_features': [batch, num_boxes, roi_height, roi_width, depth]
      'detection_classes': [batch, max_detections]
      'detection_masks': [batch, max_detections, mask_height, mask_width]
        (optional).
      'detection_keypoints': [batch, max_detections, num_keypoints, 2]
        (optional).
      'num_detections': [batch]
    output_collection_name: Name of collection to add output tensors to.
  Returns:
    A tensor dict containing the added output tensor nodes.
  """
  detection_fields = DetectionResultFields
  label_id_offset = 1
  boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
  scores = postprocessed_tensors.get(detection_fields.detection_scores)
  multiclass_scores = postprocessed_tensors.get(detection_fields.detection_multiclass_scores)
  box_classifier_features = postprocessed_tensors.get(detection_fields.detection_features)
  raw_boxes = postprocessed_tensors.get(detection_fields.raw_detection_boxes)
  raw_scores = postprocessed_tensors.get(detection_fields.raw_detection_scores)
  classes = postprocessed_tensors.get(detection_fields.detection_classes) + label_id_offset
  keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
  masks = postprocessed_tensors.get(detection_fields.detection_masks)
  num_detections = postprocessed_tensors.get(detection_fields.num_detections)
  
  outputs = {}
  outputs[detection_fields.detection_boxes] = tf.identity(boxes, name=detection_fields.detection_boxes)
  outputs[detection_fields.detection_scores] = tf.identity(scores, name=detection_fields.detection_scores)
  
  outputs[detection_fields.detection_classes] = tf.identity(classes, name=detection_fields.detection_classes)
  outputs[detection_fields.num_detections] = tf.identity(num_detections, name=detection_fields.num_detections)
  if raw_boxes is not None:
    outputs[detection_fields.raw_detection_boxes] = tf.identity(raw_boxes, name=detection_fields.raw_detection_boxes)
  if raw_scores is not None:
    outputs[detection_fields.raw_detection_scores] = tf.identity(raw_scores, name=detection_fields.raw_detection_scores)
  if keypoints is not None:
    outputs[detection_fields.detection_keypoints] = tf.identity(keypoints, name=detection_fields.detection_keypoints)
  if masks is not None:
    outputs[detection_fields.detection_masks] = tf.identity(masks, name=detection_fields.detection_masks)
  for output_key in outputs:
    tf.add_to_collection(output_collection_name, outputs[output_key])

  return outputs



def _get_outputs_from_inputs(input_tensors, detection_model,output_collection_name, **side_inputs):
  inputs = tf.cast(input_tensors, dtype=tf.float32)
  preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
  output_tensors = detection_model.predict(preprocessed_inputs, true_image_shapes, **side_inputs)
  postprocessed_tensors = detection_model.postprocess(output_tensors, true_image_shapes)
  return add_output_tensor_nodes(postprocessed_tensors,output_collection_name)


def build_detection_graph(input_type, detection_model, input_shape,
                          output_collection_name, graph_hook_fn,
                          use_side_inputs=False, side_input_shapes=None,
                          side_input_names=None, side_input_types=None):
  """Build the detection graph."""
  

  
  if input_type not in input_placeholder_fn_map:
    raise ValueError('Unknown input type: {}'.format(input_type))
  placeholder_args = {}
  side_inputs = {}
  if input_shape is not None:
    if (input_type != 'image_tensor' and
        input_type != 'encoded_image_string_tensor' and
        input_type != 'tf_example' and
        input_type != 'tf_sequence_example'):
      raise ValueError('Can only specify input shape for `image_tensor`, '
                       '`encoded_image_string_tensor`, `tf_example`, '
                       ' or `tf_sequence_example` inputs.')
    placeholder_args['input_shape'] = input_shape
  
  placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](**placeholder_args)
  placeholder_tensors = {'inputs': placeholder_tensor}
  if use_side_inputs:
    for idx, side_input_name in enumerate(side_input_names):
      side_input_placeholder, side_input = _side_input_tensor_placeholder(
          side_input_shapes[idx], side_input_name, side_input_types[idx])
      print(side_input)
      side_inputs[side_input_name] = side_input
      placeholder_tensors[side_input_name] = side_input_placeholder
  
  
  #outputs[detection_fields.raw_detection_scores] = tf.identity(raw_scores, name=detection_fields.raw_detection_scores)
  outputs = _get_outputs_from_inputs(
      input_tensors=input_tensors,
      detection_model=detection_model,
      output_collection_name=output_collection_name,
      **side_inputs
      )

    
  # Add global step to the graph.
  slim.get_or_create_global_step()

  if graph_hook_fn: graph_hook_fn()

  return outputs, placeholder_tensors



def _export_inference_graph(input_type,
                            detection_model,
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None,
                            write_inference_graph=False,
                            temp_checkpoint_prefix='',
                            use_side_inputs=False,
                            side_input_shapes=None,
                            side_input_names=None,
                            side_input_types=None):
  """Export helper."""
  tf.io.gfile.makedirs(output_directory)
  
  frozen_graph_path = os.path.join(output_directory,'frozen_inference_graph.pb')
  saved_model_path = os.path.join(output_directory, 'saved_model')
  model_path = os.path.join(output_directory, 'model.ckpt')

  outputs, placeholder_tensor_dict = build_detection_graph(
      input_type=input_type,
      detection_model=detection_model,
      input_shape=input_shape,
      output_collection_name=output_collection_name,
      graph_hook_fn=graph_hook_fn,
      use_side_inputs=use_side_inputs,
      side_input_shapes=side_input_shapes,
      side_input_names=side_input_names,
      side_input_types=side_input_types)

  profile_inference_graph(tf.get_default_graph())
  saver_kwargs = {}
  if use_moving_averages:
    if not temp_checkpoint_prefix:
      # This check is to be compatible with both version of SaverDef.
      if os.path.isfile(trained_checkpoint_prefix):
        saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
        temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
      else:
        temp_checkpoint_prefix = tempfile.mkdtemp()
    replace_variable_values_with_moving_averages(
        tf.get_default_graph(), trained_checkpoint_prefix,
        temp_checkpoint_prefix)
    checkpoint_to_use = temp_checkpoint_prefix
  else:
    checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()

  write_graph_and_checkpoint(
      inference_graph_def=tf.get_default_graph().as_graph_def(),
      model_path=model_path,
      input_saver_def=input_saver_def,
      trained_checkpoint_prefix=checkpoint_to_use)
  if write_inference_graph:
    inference_graph_def = tf.get_default_graph().as_graph_def()
    inference_graph_path = os.path.join(output_directory,'inference_graph.pbtxt')
    for node in inference_graph_def.node:
      node.device = ''
    with tf.gfile.GFile(inference_graph_path, 'wb') as f:
      f.write(str(inference_graph_def))

  if additional_output_tensor_names is not None:
    output_node_names = ','.join(list(outputs.keys())+(
        additional_output_tensor_names))
  else:
    output_node_names = ','.join(outputs.keys())

  frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=frozen_graph_path,
      clear_devices=True,
      initializer_nodes='')

  write_saved_model(saved_model_path, frozen_graph_def,placeholder_tensor_dict, outputs)

