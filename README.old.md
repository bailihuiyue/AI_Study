# AI_Study
记录下python/AI学习的笔记和一些心得

#### 人工智能简单来说就是一种人昂计算机程序能够智能的思考的方式,人工智能主要包括,机器学习和深度学习

##### 1.AI全称:Artificial Intelligence,与此对应的是NI: Natural Intelligence,指自然界的生物

##### 2.一文看懂70年的人工智能简史:<http://www.elecfans.com/d/930779.html>

​    中间经历过两次寒冬: 1975年的第一次人工智能寒冬,1990年中，人工智能又第二次遇冷
​    主要原因是算法和计算机的算力不足

##### 3.深度学习与机器学习简介<https://www.cnblogs.com/qcloud1001/p/9633724.html>

​    机器学习 ( Machine Learning ) 是一门多领域交叉学科，它涉及到概率论、统计学、计算机科学以及软件工程。机器学习是指一套工具或方法，凭借这套工具和方法，利用历史数据对机器进行“训练”进而“学习”到某种模式或规律，并建立预测未来结果的模型。
​    机器学习涉及两类学习方法：有监督学习，主要用于决策支持，它利用有标识的历史数据进行训练，以实现对新数据的标识的预测。有监督学习方法主要包括分类和回归；无监督学习，主要用于知识发现，它在历史数据中发现隐藏的模式或内在结构。无监督学习方法主要包括聚类。

![](./img/04.png) ![](./img/05.png)

##### 关于深度学习的个人理解
AI(机器学习和深度学习)和人学习一样,书读百遍,其义自见,不断地学习,理解会加深
##### 4.名词解释
######  1.名词解释:分类,聚类,回归,https://www.cnblogs.com/XinZhou-Annie/p/7253049.html
​    分类:预测值的结果是离散值: 例如，电子邮件是否为垃圾邮件，肿瘤是癌性还是良性等等。 
​    回归:预测的数据对象是连续值: 例如，温度变化或功率需求波动。 典型应用包括电力负荷预测和算法交易等。
​    聚类:用于在数据中寻找隐藏的模式或分组(物以类聚,人以群分):例如，房价预测等(百度:聚类算法应用场景实例十则)

######  2.神经元特征值,隐层,对分析结果的影响:http://playground.tensorflow.org/

   1.神经元(感知器)
![](D:\gitBackup\AI_Study\img\06.jpeg)
​     上图中每个圆圈都是一个神经元，每条线表示神经元之间的连接。我们可以看到，上面的神经元被分成了多层，层与层之间的神经元有连接，而层内之间的神经元没有连接。(模拟人的神经元)
   3.CNN,RNN,决策树等....
   4.反向传递,实际上是求导
   5.生成对抗网络,阿尔法狗



##### 5.深度学习常用框架及项目:
openai:gym,universe 通用ai框架
deepmind: alpha go ,alpha zero
udacity:自动驾驶
mujoko:仿真物理引擎,收费
roboschool:免费版mujoko
RL Lab:一个强化学习算法框架
PySC2 星际2
TensorFlow Models: tf提供的强化算法集
强化学习可以做决定

##### 6.深度强化学习分类:

在线学习(亲自参与其中):Sarsa

离线学习(看别人,从过往经验中学习):Q Learning



##### 7.些学习资料:

莫烦的知乎专栏,机器学习https://zhuanlan.zhihu.com/morvan

莫烦的github<https://morvanzhou.github.io/>

强化学习资料

Windows,Linux,macOS三平台安装OpenAI的Gym和Universe:<https://www.jianshu.com/p/536d300a397e>

什么是 Q-Learning :<https://zhuanlan.zhihu.com/p/24808797>

Flappy Bird讲解 Q - learning <https://www.zhihu.com/question/26408259>

深度强化学习相关资料:<https://www.jianshu.com/p/5ceca53aff0b>,<https://blog.csdn.net/weixin_42389349/article/details/82935123>

深度增强学习之Policy Gradient方法:<https://zhuanlan.zhihu.com/p/21725498>

强化学习效果展示https://mp.weixin.qq.com/s?__biz=MzA4NjQ4MzU4OQ==&mid=2647882155&idx=2&sn=8ba089297419799ee6148be1a0a53395

DeepLearningBook读书笔记<https://github.com/exacity/simplified-deeplearning>

人工智能之机器学习常见算法:<https://www.cnblogs.com/xiyushimei/p/7874019.html>

深度学习之美:<https://yq.aliyun.com/articles/86580>

外语学习:西雅图工作英语

数学学习:数学之美



关于项目中的代码:

RNN循环神经网络 文件夹中的部分demo来自慕课网,我增加了注释和一些其他东西
其中

1.midi音乐创造使用的是RNN循环神经网络

2.P图使用的是DCGAN,深度卷积对抗网络 

##### 注:ai p图的demo需要pip install pillow

3.自动玩游戏使用的是强化学习(深度强化学习)RL,

其中最后一个赛车demo由于windows不支持安装环境,所以代码没有实际测试,

以上项目都是使用TensorFlow1.x版本

##### 