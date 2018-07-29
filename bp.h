//bp.h
#ifndef _BP_H_
#define _BP_H_
#include <vector> //这是C++库里的一个向量类 
#define LAYER 3 //设置神经网络为3层，因为C++的脚标是从0开始的,所以即是：0、1、2 
#define NUM 10 //每层节点数最多10个，即0~9 
#define A 30.0  
#define B 10.0 //在这里我将采用广义的sigmoid函数作为激活函数，广义的sigmoid函数
//相对于狭义的其实就是多了两个常数参数，如果你想把激活函数退化为狭义的sigmoid，
//那你只需要把A，B都设置为1就行，我在这儿就先搞个经典的广义sigmoid函数
#define ITERS 1000 //默认训练次数
#define Type double //这些都是宏定义，意思就是以后见到Type就默认当成double,我这里之所以要这样定义一个
//宏，相当于把Type在之后用作double是为了提高代码的可重用性，因为这样下次你想处理整数时，只要在这里把
//double换成int就行了 
#define Vector std::vector //vector是C++里面内置的额一个模板类。 
//把具体的类型给它，这里给了它double类型，这个模板就具体化为一个类型为double的具体类了。 
struct Data
{
    Vector<Type> xxx; //这个X是一个具体类型为double的vector对象，之后用来储存input nodes
	Vector<Type> yyy; //这个y则是之后用来储存output nodes  
};//在此，我要特别感谢C++标准库里内置的这个vector类模板，因为它是智能的：你要生成一个Vector向量对象，仅仅
//需要做的就是告诉它你选择的数据类型，而不用告诉它你要创建一个多少位的向量，你输入几个数，Vector类内部就会
//做相应记录并把这个Vector对象设置为这一对应长度。这一特点，在本例中将给我们以很大的好处。即当我们输入不同
//长度的向量时，不用一次次先去提前修改数据类型。只要你这次输入的个数和上次不一样了，模板就自动察觉到，并会
//给你生成新的长度的向量容器。 
/*所以当本例中我们需要修改网络的结构时，如从（3,3,1）改到（6,6,1）时，X需要从一个长度为3的向量改为一个长度
为6的向量，但这点是不用我们手动操作了，你每次给X输入几个数，他就会自动生成为一个长度为几的向量。所以当我们
变更网络结构时，只要注意匹配地做好两件事就行：1.手动输入新的pair，2.把服务器外部主函数里的那个"_data"的每
个小片段从（3,1）改为（6,1）即可*/ 

//以上声明了一个叫Data的结构，这个结构是用来装“一份最小的样本单元”的，它里面就是两个double向量罢了
//比如，当我们配置神经网络为（3,3,1）时，那么所谓“最小的一份样本unit”就是三个输入值，一个用于验证的输出值。
//所以相应的Data类对象就是两个向量：一个（x1,x2,x3),一个（y1),总之， Data类型的结构就是一个用来储存
//“一份最小样本单元”的容器 
//下面声明一个叫BP的类 
class BP{  
public:  
    void GetTrainData(const Vector<Data>); //即这个GetTrainData()函数的参数是一个vector向量类的对象，这个每一维度都是一个Data类的对象
	                //换句话说，Vector<Data>类的对象是这样一种东西：本身是一个向量，而这个向量的每一维里存的是一份
					//“最小样本单元”（即两个double向量），所以整个Vector<Data>对象就是一系列的“最小样本单元”                   
	void GetValiData(const Vector<Data>);//上面那个是读入训练数据集，这个是读入验证数据集 
	void SetNodes(int in=3,int hd=3,int ou=1);//获取输入层、隐含层、输出层的节点数。也就是获取这个网络的结构
	void SetParas(double lr,double mt);//获取学习率和动量项的函数 
	void InitNetWork();          //初始化网络 
	void Train();	
	Type GetValiAccu();          //计算已经经过训练的网络,应用于验证数据集时产生的平均方差(MSE) 
	Vector<Type> ForeCast(const Vector<Type>); //这个forecast函数接受一个类型为vector-double的向量类对象 
 //作为参数，它的返回值类型也是一个vector-double的向量类的对象
   
private:
	void ForwardTransfer();      //正向传播子过程
	Type GetError(int);          //计算“一份最小样本单元”运算一次后于target value的误差
	void ReverseTransfer(int);   //逆向传播子过程
	void CalcDelta(int);         //计算权重向量Wij的调整量，里面用到了学习率和动量项
	void UpdateNetWork();        //更新权值向量 
    Type Sigmoid(const Type);    //计算sigmoid的值，即激活一个神经元，并求出它的输出 
 
private:   
	int in_num;                  //输入层的节点数,即input nodes数目 
	int hd_num;                  //隐藏层的节点数，即hidden layer nodes数目 
	int ou_num;                  //输出层的节点数,即output nodes数目，在本例中，我们总是取1 
    double Lerate;               //这是学习速率，learning rate
    double Moterm;               //这是动量项，momentun term
    
    Vector<Data> tdata;           //用来装training时输入的数据。
	                             //这个data变量就是上面详细说过的Vector<Data>类的一个具体的对象了。它最外层是一个向量的结构，
	                             //但是这个向量的每一维都不是一个简单的double值什么的，而都是一个Data类型的变量。
								 //即这个data向量，它的每一维都是一份“最小样本单元”。整个data就是一系列的“最小样本单元” 
    Vector<Data> vdata;          //用来装validate时输入的数据 
	Type w[LAYER-1][NUM][NUM];   //这个三维数组w就是整个BP网络的权重立方阵了，这是按照最大容量设计的,即可以装下结构为（10,10,10)的
	                             //BP网络的权重立方阵，而我们最大的也才只有（10,10,1），够用了。另外，对于三层的神经网络，权重
								 //矩阵只有两个，即输入层和隐藏层之间一个，隐藏层和输出层之间一个，所以W的第一个维度只有2 
    Type Lw[LAYER-1][NUM][NUM];  //Lw中的L是延迟算子，所以Lw表示上一次的w阵，即,如果说w表示w(t)的话，那么Lw就表示w(t-1) 
	Type x[LAYER][NUM];          //X用来装整个神经元共三层的输出值，其中输入层的输出是不经过激活函数处理的，所以输入层的输出值就是输入的原值
	                             //只有隐藏层和输出层的输出才是将该神经元的输入值经过激活函数处理一下。同样的，我也是按最大容量设计的
								 //所以对于一个三层的神经网络，需要三个向量分别来装每一层的输出向量，X就是这三个向量的合阵
								 //所以X的第一维个数是3，这一点与W不同，要注意 
	Type d[LAYER][NUM];          //d就是指delta，我们知道最后的Error对W(i,j)的偏导数可以写成delta(i,j)*Xi,其中Xi表示这个W所联系的两层
	                             //中的前一层中第i个神经元的输出值。而delta(i,j)其实有很好的性质：即它和i没有关系其实。它只是一个关于j
								 //的函数。所以对于一个（3,3,1）的网络来说，它的第输入层和隐藏层直接共有3*3=9个W(i,j)，也确实每个W(i,j)
								 //都应该对应一个delta(i,j)的，但是却只有3个不同值的delta，而不是9个，因为比如对于隐藏层的第二个节点，则
								 //j=2，那么delta(1,2),delta(2,2),delta(3,2)这三个是一样的，因为delta(i,j)之和j有关。所以这里设置变量的时候
								 //虽然理论上delta(i,j)和W(i,j)应该是一一对应的，但是实际用的时候，W比delta多一维 
};

#endif                

//这个是头文件，是对我们之后要用到的所有工具的简单描述，这些工具具体是怎么样的、他们如果工作，将在下一个
//文件中说明，即这里声明的函数和类的具体代码实现都放在另一个源代码文件里了
//这个头文件和另一个代码文件，共同构成一个服务器。这个服务器是被封装好的，从此可以处理任意地数据集 

	                      
 
 
 
 
 
  
