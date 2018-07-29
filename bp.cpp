//bp.cpp
#include<string.h>
#include<iostream>
#include<math.h>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<assert.h>
#include"bp.h"
using namespace std;

void BP::GetTrainData(const Vector<Data> _tdata){
	tdata=_tdata;  
}
//这个函数是用来将用户的数据正式地读入到我写好的神经网络服务器里的 
//也就是说在服务器之外，在main函数中，已经有一个叫做 "_data"的vector对象了，它里面装了一系列的“最小样本单元” 
//然后这个函数的功能是将这一系列的最新一部单元赋值给BP这个类中定义的一个叫data的Vector<Data>容器总，其实类中
//的data容器和类外主函数里的_data容器是两个同样规格的容器 

void BP::GetValiData(const Vector<Data> _vdata){
	vdata=_vdata;
}

void BP::SetNodes(int in,int hd,int ou){//这个BP类的公有成员函数的作用是：把服务器外的三层的节点数读入到 
	in_num=in;                              //到服务器中，以供之后在服务器里生成一个具体规格的神经网络 
	hd_num=hd;
	ou_num=ou;
}

void BP::SetParas(double lr,double mt){
	Lerate=lr;
	Moterm=mt; 
}

void BP::InitNetWork(){//这个成员函数的作用是：根据神经网络的具体规格，生成初始化为随机数的权重立方阵 
	srand(time(NULL));
    memset(w,0,sizeof(w));//先把整个用来装权重矩阵的容器W都初始化为0,然后再逐项赋值一个随机数 
	memset(Lw,0,sizeof(Lw));//这个是W阵的一步延滞阵 
	for(int i=0;i<in_num;i++){
		for(int j=0;j<hd_num;j++){
		   
			w[0][i][j]=1.0*rand()/RAND_MAX-0.5;//这是在产生(-0.5,0.5)的随机数，用这样的随机数填充输入层和隐藏层之间的权重矩阵 
            Lw[0][i][j]=w[0][i][j];	
            
		} 
	} //这里有一点要注意的：W的第一个维度就是layer，由于C++的数组脚标是从0开始的，所以第一维是0就表示第一个layer，即 
    for(int i=0;i<hd_num;i++){//输入层和隐藏层之间的那个权重矩阵layer，而第一维是1就表示第二层，即隐藏层和输出层之间的 
    	for(int j=0;j<ou_num;j++){//那层权值矩阵 
		    
    	    w[1][i][j]=1.0*rand()/RAND_MAX-0.5;//这是用(-0.5,0.5)的随机数填充（初始化）隐藏层和输出层之间的权重矩阵 
		    Lw[1][i][j]=w[1][i][j];//在初始化的时候把w和Lw两个立方阵取成完全一样的。这样做的意义是：在第一次学习之前，	
		
		}//增量位0 
	}
}

void BP::Train(){//千万记住，在训练网络之前要先设置网络的具体结构并初始化网络！！！ 
	cout<<"Begin to train BP Neural Network!\n"; 
	int num=tdata.size(); //之前已经说过，data变量就是一系列的“最小样本单元”，它本身属于Vector类对象。
	                      //Vector类是自带了一个size()函数的。那么这个data变量的size就是这“一系列最小样本单元”
						  //里到底有多少份这种“最小样本单元”，比如在（3,3,1）的网络结构下，有70%*260=182个数据
						  //用于学习，把这182个数据制作成“3个输入值，1个输出值”的“最小样本单元”的话，共
						  //做成179份，那么在这样的设置下，这个data变量的size就是179，因为它含了179份“最小样本单元” 
	
    for(int iter=0;iter<ITERS;iter++){//这一层循环，循环一次就是完成了整个training，就是产生3000个epoch 
    	for(int cnt=0;cnt<num;cnt++){//这一层循环，循环一次就是产生一个epoch，也就是把179份“最小样本单元”都过一遍
		                  //事实上，其中的num就是data的size，在我们的例子中num=179，cnt将从0遍历到178，总共也是179次 
    		for(int i=0;i<in_num;i++){x[0][i]=tdata.at(cnt).xxx[i];}//data.at(cnt)表示取data这个Vector向量的第cnt维度 
    		/*这个at()函数是Vector类自己定义的一个公有函数，即取data这个复合向量里的第cnt份“最小样本单元”，
			这样取出来的东西就是一个Data类型的结构。然后取这个Data变量中的x向量，并取这个x向量的第i个值*/ 
		    ForwardTransfer();  //顺的捋一次 
			ReverseTransfer(cnt); //倒的捋一次，更新一次权重向量 
		}  //至此，一个epoch产生完毕  
	} //至此，对于给定结构的神经网络，比如（3,3,1），3000个epoch已经产生，就这个特定结构的网络而言，学习结束 
//	double TrainingFinalError=GetError(num-1); //把3000个epoch的最后一个epoch的最后一小片基因的学习后误差作为经历了整个学习
	                                 //过程（即179*3000次往复）后的误差 
 	cout<<"Training has been completed!\n";
//<<" The final trianing squared error of this "<<ITERS;
//	cout<<" epoches is: "<<GetError(num-1)<<endl;
//	cout<<"Please NOTE that this is just the final error after training set.\n";
//	cout<<"We haven't applied this training result on the validation set yet.\n";
} 

void BP::ForwardTransfer(){
	for(int j=0;j<hd_num;j++){
		Type s1=0;
		for(int i=0;i<in_num;i++){s1+=w[0][i][j]*x[0][i];}//s表示隐藏层每个节点的输入值，在本例中我们不考虑阀值和偏移值
		                                                 //每个x[0][i]的值则是Train()函数在调用本函数之前就会先设置好的 
	x[1][j]=Sigmoid(s1);//这就是隐藏层的每一个节点的输出值了 
	}
	
	for(int j=0;j<ou_num;j++){//其实在我们的例子中，ou_num恒等于1,但是我写成这样的广义形式，则这个代码不仅可以用于基于 
	    Type s2=0;            //过去的学习对未来做一步预测，还可以对未来做多步预测 
		for(int i=0;i<hd_num;i++){s2+=w[1][i][j]*x[1][i]; }
    x[2][j]=Sigmoid(s2);   	    
	} 
}

Type BP::GetError(int cnt){
	Type ans=0;
	for(int i=0;i<ou_num;i++){
		ans+=0.5*(x[2][i]-tdata.at(cnt).yyy[i])*(x[2][i]-tdata.at(cnt).yyy[i]);
	}
	return ans;
} 

Type BP::GetValiAccu(){//这个函数的作用是将在training set上训练好的网络应用于一个validation set，将 
	Type ans=0;        //validation set上的每一份“最小样本单元”依次放入网络中往前捋一遍，得到结果后 
	int num=vdata.size();//求出误差的平方和，最后求平均的误差平方和 
	for(int i=0;i<num;i++){
		int m=vdata.at(i).xxx.size();
		for(int j=0;j<m;j++){
		    x[0][j]=vdata.at(i).xxx[j];
		}
		ForwardTransfer();
		int n=vdata.at(i).yyy.size();
		for(int j=0;j<n;j++){
		    ans+=0.5*(x[2][j]-vdata.at(i).yyy[j])*(x[2][j]-vdata.at(i).yyy[j]);
		}
	}
	cout<<"validation error calculated!\n";
	return ans/num;
}

Vector<Type> BP::ForeCast(const Vector<Type> fdata){//注意，这里的形参名字也叫data，但这里只是一个零时性的称呼 
                                                   //而已，和上文中Train()函数里的成员变量data是两回事儿,并且
												   //这里的这个data只是一个double类型的vector，而Train()函数里的
												   //成员变量data则是一个Data类型的vector 
	int n=fdata.size();
    assert(n==in_num);// 检查非预期状态的最简单的方式是通过标准C库的assert宏。这个宏的参数是一个布尔
	                  //表达式。当表达式的值为假的时候，assert会输出源文件名、出错行数和表达式的字面内容，
					  //然后导致程序退出。Assert宏可用于大量程序内部需要一致性检查的场合
	    //可见，ForeCast()函数的所接受的参数就只是一个规格和“一份最小样本单元”一样的单个基因，不能接受连续的
		//基因片段作为参数 
	for(int i=0;i<in_num;i++){x[0][i]=fdata[i];}
	ForwardTransfer();
	Vector<Type> v;
	for(int i=0;i<ou_num;i++){v.push_back(x[2][i]);}//介绍一下这里用到的push_back()函数,c.push_back(X)的意思就是把元素X加入 
	return v;	 //到容器c的最后一位，像这里容器是一个vector对象，我们只能用push_back函数，没法用等号赋值 
}//这个函数是用来预测的，但是，它只是功能最简单的“单基因”预测，比如（3,3,1）的神经网络 ，那么这个预测函数就是接受
 //一个3维的double向量作为参数，然后输出一个1维double向量（也就是一个double值的数）作为输出。
 //这样看来，似乎这个 ForeCast函数只能做一步预测， 但其实也未必，因为多步预测其实有两种实现方法：第一是每次ForwardTransfer
 //只往前预测一步，然后一次ForeCast中ForwardTransfer多次，第二是每次ForwardTransfer往前预测多步，然后
 //一次ForeCast就ForwardTransfer一次。当然，这两种方法也并不是等价的，因为所用到的信息量不一样。
 //anyway，在本例中，由于作者只用ForwardTransfer往前坐一步预测，而这个ForeCast函数中又仅仅使用了ForwardTransfer，所以这个
 //ForeCast函数在本例中实际用处其实就是像本段第二行说的那样，最后预测出一个double值的未来值。所以它的实际引用场合也就是
 //最后最后一切都做完了之后在stage3里去预测一下那一个“明天的值” 

void BP::ReverseTransfer(int cnt){
	CalcDelta(cnt);
	UpdateNetWork();
}

void BP::CalcDelta(int cnt){
	for(int j=0;j<ou_num;j++){
		d[1][j]=(x[2][j]-tdata.at(cnt).yyy[j])*x[2][j]*(A-x[2][j])/(A*B);
	} //这里d的第一维和W的第一维是对应的 
	for(int i=0;i<hd_num;i++){
		Type t=0;
		for(int j=0;j<ou_num;j++){
			t+=d[1][j]*w[1][i][j];
		}
		d[0][i]=t*x[1][i]*(A-x[1][i])/(A*B);
	} 
}

void BP::UpdateNetWork(){
	//隐藏层和输出层之间的权重调整：
	Type temp; 
	for(int i=0;i<hd_num;i++){
		for(int j=0;j<ou_num;j++){	
		    temp=w[1][i][j];
			w[1][i][j]=w[1][i][j]-Lerate*d[1][j]*x[1][i]+Moterm*(w[1][i][j]-Lw[1][i][j]);
		    Lw[1][i][j]=temp;//用temp来跟新Lw,使它始终与w保持一步延滞 
		}
	}
	//输入层和隐藏层直接的权重调整：
	for(int i=0;i<in_num;i++){
		for(int j=0;j<hd_num;j++){
			temp=w[0][i][j];
			w[0][i][j]=w[0][i][j]-Lerate*d[0][j]*x[0][i]+Moterm*(w[0][i][j]-Lw[0][i][j]);
			Lw[0][i][j]=temp;
		}
	} 
}

Type BP::Sigmoid(const Type x){
	return A/(1+exp(-x/B));
}//在本例中，我们采用了这个S型激活函数 

//至此，我们实现了头文件中的所有函数 
