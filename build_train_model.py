import tensorflow as tf
import os


if __name__ == '__main__':

    # 1. 定义模型
    inputs = tf.keras.Input(shape=(32,) ,name='input_new' )
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='output_new')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    # 2. 编译模型
    model.compile(optimizer=tf.optimizers.Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])


    # 4. 输出模型
    tf.saved_model.save(model, './DIYmodel')

    # 5. 加载模型
    loaded_model = tf.saved_model.load('./DIYmodel')

    # 6. 修改输出名称
    #loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].output_aliases["output_layer"] = "score"

    # 7. 保存修改后的模型
    #tf.saved_model.save(loaded_model, './DIYmodel/modified_saved_model')
