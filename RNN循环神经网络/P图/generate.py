# -*- coding: UTF-8 -*-

"""
用 DCGAN 的生成器模型 和 训练得到的生成器参数文件 来生成图片
 生成对抗网络(GAN)的原理
 GAN的主要灵感来源于博弈论中零和博弈的思想，应用到深度学习神经网络上来说，就是通过生成网络G（Generator）和判别网络D（Discriminator）不断博弈，进而使G学习到数据的分布，如果用到图片生成上，则训练完成后，G可以从一段随机数中生成逼真的图像。G， D的主要功能是：

 G是一个生成式的网络，它接收一个随机的噪声z（随机数），通过这个噪声生成图像 ；

 D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片

 训练过程中，生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量辨别出G生成的假图像和真实的图像。这样，G和D构成了一个动态的“博弈过程”，最终的平衡点即纳什均衡点.https://www.cnblogs.com/eilearn/p/9490288.html
"""

import numpy as np
from PIL import Image
import tensorflow as tf

from network import *


def generate():
    # 构造生成器
    g = generator_model()

    # 配置 生成器
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))

    # 加载训练好的 生成器 参数 TODO:既然有训练好的参数可以使用了,为什么还需要上面那句话配置生成器和优化损失呢(难道是为了在生成的过程中二次优化?)
    # 答案:这种方法不会保存整个网络的结构，只是保存模型的权重和偏置，所以在后期恢复模型之前，必须手动创建和之前模型一模一样的模型，以保证权重和偏置的维度和保存之前的相同。https://blog.csdn.net/wwwlyj123321/article/details/94291992
    # tf.keras.model类中的save方法和load_model方法,这种方法会将网络的结构，权重和优化器的状态等参数全部保存下来，后期恢复的时候就没必要创建新的网络了。
    g.load_weights("generator_weight")

    # 连续型均匀分布的随机数据（噪声）
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    # 用随机数据作为输入，生成器 生成图片数据, 
    images = g.predict(random_data, verbose=1)

    # https://www.imooc.com/article/28569
    # TODO:上面两句不是很理解,如果random_data随机好了,岂不是每次图片就靠随机值生成了?,那还要load_weights有啥用 
    # 查看几篇文章发现,训练时就是生成一张张噪点,然后用噪点(随机值)去生成图片,通过多次的训练随机值就会越来越趋近于真实值(为啥),最后生成图片时传入一个随机值,就能生成一张图片,问题是随机值怎么变成的图片,为啥不直接生成有意义的值或者图片
    
    # 猜测:生成的随机数就好比一幅画或者图片的骨架或者轮廓,ai可以通过学习到的weights来画画,把随机值能成的轮廓画成一幅画

    # 用生成的图片数据生成 PNG 图片
    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 + 127.5 #反标准化
        Image.fromarray(image.astype(np.uint8)).save("image-%s.png" % i)


if __name__ == "__main__":
    generate()
