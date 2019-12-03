## 初学BERT的一点思考
---

&#160; &#160; &#160; &#160;此前一直没认真看过`bert`，今天看源码的时候发现`Position embedding`居然是随机初始化的可训练的参数，有点惊讶，毕竟`transformer`里是三角函数确定的常值，通过和差化积体现的相对位置关系。  

&#160; &#160; &#160; &#160;因此，我不禁就很疑惑这个`PE`训练后**真地的是提供的相对位置信息吗**？  

&#160; &#160; &#160; &#160;于是，我先看了下`PE`权重的直方图，基本跟初始化一样，还是标准差为`0.02`的截断正态分布，超出两个标准差的参数（训练的结果）很少，这个还算正常，本来网络中大部分参数用处都不大。  
![权重分布直方图](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/distriution.jpg)  
&#160; &#160; &#160; &#160;然后我画了个`heatmap`，把两个标准差以内的变`0`，两个标准差外的分别变为`-1`和`1`。这么看，好像有点意思，**`2`个标准差外的值主要都在`pos`比较靠前的部分**。这个地方我想了想**可能跟实际训练中，靠后部分主要都是`\[PAD\]`有关**。  
![初始热图](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/Positional_Embedding_value.jpg)  
&#160; &#160; &#160; &#160;不过很明显，两个标准差外的值也不是在所有`embedding dimension` 出现，考虑到**本身这些`dimension`就不一定独立**，并且各个**`dimension`是无序的，即等价**的。于是我就想先搞个**主成分提取**，解释度高于`0.01`的标出来了。  
![主成分解释度](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/component_ratio.jpg)  
&#160; &#160; &#160; &#160;我先尝试用最主要的两个成分可视化了一个散点图，以为会出现啥螺纹曲线的，结果这个基本没啥用，确实可能`2`个维度太少了，不足以反映相对位置关系。  
![PCA 2-D](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/PE_PCA_2D.jpg)  
&#160; &#160; &#160; &#160;然后我想看看各个正交的主成分随`POS`变化的图像，就做了以下四张图，因为`\[CLS\]`永远都在第一位，所以分别是带不带第一个`\[CLS\]`符的`PCA`，想看看差别。前七个主成分很接近，靠后的一些也很接近，但是两者在第`8、9`/`9、10`两个差别还是蛮明显的，**这个地方，从矩阵秩的角度，我似乎有那么一点点意会，但没想太清楚**。  
![带CLS 1-9](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/pos_component_value_with_cls_1-9.jpg)  
![带CLS 10-18](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/pos_component_value_with_cls_10-18.jpg)  
![不带CLS 1-9](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/pos_component_value_without_cls_1-9.jpg)  
![不带CLS 10-18](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/pos_component_value_without_cls_10-18.jpg)  
&#160; &#160; &#160; &#160;然后从这些图来看，倒还是很接近三角函数的，尤其是靠前的`POS`（感觉也是同样的原因，在实际训练中，靠后可能主要都是`\[PAD\]`），这说明随机初始化可能**确实学到了相对位置信息**。下面主成分做得`heatmap`。  
![PCA 热图](https://github.com/Lintianqianjin/NLP-models/blob/master/BERT/PE_IMG/Positional_Embedding_value_PCA_T.jpg)  
&#160; &#160; &#160; &#160;如果能真地反映相对位置关系，那我应该可以**反过来通过两个不同`POS`的`embedding`推测两者的距离**，例如`pos_a`是`5`，`pos_b`是`7`，`pos_c`是`4`，那么`b`,`c`相对`a`的距离分别是`2`，`-1`。于是我就不管`\[CLS\]`的`PE`，用余下`511`个`PE`，构建了一个数据集，任意两个位置之间的构成一个样本（有向），共`261121`个样本，分为了`0.6`: `0.4`的训练集和测试集。刚开始我尝试用神经网络去拟合这个函数关系`***f(pos_a_emb,pos_b_emb) = pos_b_index - pos_a_index***`，设计了一些比较复杂，也设计了一些比较简单的网络，可能还是网络没设计好，总之尝试的模型结构，`loss`都不下降。  

&#160; &#160; &#160; &#160;但是根据图像，我想数据本身应该还是能反映一点相对位置信息的，我决定用传统机器学习的一些回归算法先试试，看看**是否存在优化的可能性**。输入是两个`POS` `embedding`的`18`个主成分，最开始用线性回归，测试集均方误差很高，在`9000+`，后来用`GBDT`，`min_leaf_nodes = 15`，**测试集均方误差能到`1800`，均绝对误差`30`**，应该说提高了很多，肯定是好于盲猜了。然后我想既然是用`GBDT`，那我直接用原始`768`维的向量当作输入试试，结果误差又变得很高（**结合第一张`HEATMAP`看，确实很多维在位置上是没有意义的，这样应该是引入了噪音**）。  

&#160; &#160; &#160; &#160;我想的话，下一步，是把`POS`限定在`128`以内，避免在训练中加到`\[PAD\]`上的`PE`产生干扰，可能回归结果会好一些。  
&#160; &#160; &#160; &#160;另外，**到底是为什么能学到相对位置信息呢**？  
