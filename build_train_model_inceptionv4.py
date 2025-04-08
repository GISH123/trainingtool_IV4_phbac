import os

import tensorflow as tf
from models.inception_modules import Stem, InceptionBlockA, InceptionBlockB, InceptionBlockC, ReductionA, ReductionB
from configuration import NUM_CLASSES
from prepare_data import load_and_preprocess_image
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


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        
        #self.dense_1 = tf.keras.layers.Dense(10)
        #self.dense_2 = tf.keras.layers.Dense(1)
        #self.dropout = tf.keras.layers.Dropout(0.1)

        
        self.stem = Stem()
        self.inception_a = build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
      
        
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax  ,name='score' )
        
       

    def call(self, inputs, training=None):
        #a, b = inputs
        a = inputs
        
        #hidden = self.dense_1(a + b)
        #hidden = self.dropout(hidden, training=training)
        #logits = self.dense_2(hidden)


        x = self.stem(a, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        x = self.fc(x)
        x = tf.identity(x, name="new_output")
        logits =x 
       
        return logits 

    @tf.function(input_signature=[(
        
                                tf.TensorSpec([1, 299,299,3], name='image_tensor', dtype=tf.float32)
                                
                                #tf.TensorSpec([1, 299,299,3], name='tensor_a', dtype=tf.float32),
                                #tf.TensorSpec([None, 10], name='b', dtype=tf.float32)
                                   
    )])
    def sever(self, inputs):
        output=self.call(inputs, training=False)
        return {"score": output}
    
    

    def __repr__(self):
        return "InceptionV4"

if __name__ == '__main__':
    model = MyModel()
    #data1 = tf.ones((2, 10))
    #data2 = tf.ones((2, 10))
    #out = model((image_tensor, data2), training=True)
    

    test_image_dir='testone/c1.jpg'
    image_raw = tf.io.read_file(filename='./'+test_image_dir)
    image_tensor = load_and_preprocess_image(image_raw)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
   
    out = model((image_tensor), training=True)
    print(out)
    #model.save("./DIYmodel", signatures={"serving_default": model.sever})
    model.save("./DIYmodel")
    


