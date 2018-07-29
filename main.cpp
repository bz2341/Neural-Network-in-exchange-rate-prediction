/*注意：
以下这个代码模型有两个可以改良的地方
1.尚未对每个给定结构的网络使用蒙特卡洛方法来充分测试不同的随机初始权重立方阵 
2.即便是像现在这样不对每一个具体结构的网络使用蒙特卡洛模拟，其实也可以尝试让这些不同结构的网络的初始随机权重立方阵互不相同，在本例中，它们则基本是相同的 
有一个使用时要注意的地方：只能读取260个数据，即用户那个作为数据来源的Excel表格最好是干干净净的就260个数据，别的杂物不要有，并且数据要不多不少刚好260个*/
#include<iostream>
#include<fstream>//使用文件输入输出流
#include<string>
#include<string.h>
#include<stdio.h>
#include<cstdlib>
#include<ctime>
#include<cmath> 
#include"bp.h"
using namespace std;

Vector<Data> TrDatArg(double* r,int in,int out,double TrPortion){//这个函数名的意思即training data arrange 
                         	//这个函数专门负责把一整串的全年数据或者多年数据处理成特定结构-比如
							 //(3,3,1)-的神经网络能直接接受的数据形式，即做成“一系列的基因片段” 
	int size=260;
	int num=(int)(size*TrPortion-in);//这个num并不是用来记录training set将中的数据数量，而是记录
	                                //training set中将含有的“最小样本单元”的份数                                 
    Vector<Data> _tdata;
    for(int i=0;i<num;i++){
    	Data t;
    	for(int j=i;j<in+i;j++){t.xxx.push_back(r[j]);}
	    for(int j=0;j<out;j++){t.yyy.push_back(r[i+in+j]);}
	    _tdata.push_back(t);
	}          
	return _tdata;
}//至此，_tdata产生 

Vector<Data> ValiDatArg(double* r,int in,int out,double TrPortion,double ValiPortion){
	/*int size=0; 
	while(r[size]!=10000){                
		size++;
	}*/
	int size=260;
	int n=(int)(size*TrPortion);//这前n个是training用的数据集，从这第n个开始才是validate用的数据 
	int num=(int)(size*ValiPortion-in);//这是计算validate用的“最小样本单元”的份数 
	Vector<Data> _vdata;              
    for(int i=0;i<num;i++){
    	Data t;
    	for(int j=i;j<in+i;j++){t.xxx.push_back(r[n+j]);}
	    for(int j=0;j<out;j++){t.yyy.push_back(r[n+i+in+j]);}
	    _vdata.push_back(t);
	}          
	return _vdata;
}//至此，_vdata产生 

int main(){
	ifstream is;
	is.open("c:\\file dock\\exchange\\ExchangeRateData2016.csv");
	if(!is.is_open()){       
	cerr<<"打开文件失败！\n"; 
	return -1;
    }
    double rawdata[260];
    for(int i=0;i<260;i++){
    	is>>rawdata[i];
	}
    is.close();//至此，已经将csv文件中的所有原始数据读入到了一个叫rawdata的double数组中了 
    
    cout<<"these are the rawdata:\n";
    for(int i=0;i<260;i++){
    	cout<<rawdata[i]<<endl;
	}

    
   //以下是曾经的隔离检查区域！！！！！！！！！！！！！
   //我不想删掉这行注释，因为为了纪念这个曾经被当作bug的新特性的繁琐验证过程。正如鲜艳的西红柿在中世纪  
     
        
    
//先来解决第一个问题，即给定learining rate=0.3、momentum term=0.3，让我们选择最优的(input nodes, hidden nodes)pair
//其实就是逐一测试从(3,3,1)到(10,10,1)的所有这64种结构的神经网络，看在一样的材料和一样强度的training下，哪个模型最后完成得最好： 
    double error1[64]; int i=0; 
	Vector<Data> _tdata, _vdata; //这就是两个容器，声明了以后下面就可以一直用 
    
	for(int in=3;in<=10;in++){
    	for(int hd=3;hd<=10;hd++){
    		_tdata=TrDatArg(rawdata,in,1,0.7);
    		_vdata=ValiDatArg(rawdata,in,1,0.7,0.1);//至此，在任何一种结构配置下都准备好了两份可供网络直接使用的数据：训练数据和验证数据 
    		BP bp1;//这个bp1是在内层循环里声明的，所以它是个局部变量而已，一旦这个循环结束了，这个变量就会消亡 
		    bp1.SetNodes(in,hd,1);
			bp1.SetParas(0.3,0.3);
			bp1.GetTrainData(_tdata);
			bp1.InitNetWork();
			cout<<"For (inputNodes, hiddenNodes, outputNodes)=("<<in<<','<<hd<<",1) network: \n";
			bp1.Train();
			bp1.GetValiData(_vdata);
			cout<<endl;
			error1[i]=bp1.GetValiAccu(); 
			i++;//所以error1[0]~erroe1[7]这8个就是(3,3,1)~(3,10,1)这八个网络的validation error 
		}//i的取值是0~63 
	}//至此，我们把从(3,3,1)到(10,10,1)这64种网络都试了一遍，并得到了它们在相同学习强度下各自得到的validation时候的误差，都储存在error1这个数组里 
    //然后我们来找出error最小的那对pair：
    int j=0;
	for(int i=0;i<64;i++){
//		cout<<"error1["<<i<<"]="<<error1[i]<<endl;
		if(error1[i]<error1[0]){
			error1[0]=error1[i];
			j=i;
		}
	}//这个方法能够使得error1中最小的一个数被放在error1[0]的位置并且它原来的脚标被储存在j中,现在我们来还原回这个j对应的pair：
	int best_hd=(j%8)+3;
	int best_in=(j+1-(j+1)%8)/8+3;
    cout<<"The BEST pair of input nodes and hidden layer nodes is: ("<<best_in<<','<<best_hd<<").\n";
	cout<<"And the validation mean square error(MSE) is: "<<error1[0]<<endl;    
    //至此，完成了STAGE 1的任务！
 
    cout<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------Stage 1 complete-----------------------------"<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;


//接下去解决第二个问题，即给定输入层和隐藏层的节点数为最佳的以后，让我们选择出最优的(learing rate,momentun term)pair:
//则类似地，逐一测试(learing rate,momentun term)从(0.1,0.1)到(0.9,0.9)这81个pair，看在一样的材料和一样强度的training下，哪个模型最后完成得最好：
    
	double error2[81]; i=0;//在问题1的解决过程里声明的变量i仍旧可以用，不要浪费
    _tdata=TrDatArg(rawdata,best_in,1,0.7);//现在，网络的结构是一直固定了的，就都是(best_in,best_hd,1)。并且由于网络的结构定了，所以它们
    _vdata=ValiDatArg(rawdata,best_in,1,0.7,0.1);//接受的“一系列基因片段”的形态也就定了，所以不用像上一个问题那样每次都重新生成训练材料和验证材料 
	for(double lr=0.1;lr<=0.9;lr=lr+0.1){
		for(double mt=0.1;mt<=0.9;mt=mt+0.1){
    		BP bp2;//这个bp2和上面的bp1一样，我把它们处理成局部变量，而不是在循环外先声明一个全局变量，然后每次循环都仅仅修改其参数，是因为那样就像要做一组实验
			       //但只拿了一个容器，每次实验以后就把这个容器洗洗再用。而我现在这样做，就像做这一组实验中的每一个时，都去拿一个一次性的容易，这次实验做完
				   //这个容器就扔掉。这样做的好处是比较可靠，每次实验开始时容器都是全新的、干净的，不会因为上次实验的残留没洗干净而导致对这次实验的影响 
		    bp2.SetNodes(best_in,best_hd,1);
			bp2.SetParas(lr,mt);//现在这个问题里正真变化的是这个pair！之前是用固定的(0.3,0.3) 
			bp2.GetTrainData(_tdata);
			bp2.InitNetWork();
			cout<<"For (learning rate, momentun term)=("<<lr<<','<<mt<<") network: \n";
			bp2.Train();
			bp2.GetValiData(_vdata);
			cout<<endl;
			error2[i]=bp2.GetValiAccu(); 
			i++;
		}
	}//至此，就以相同的学习强度遍历了所有我们要检测的81个learning rate和momentum term的pair，并把每一对在相同验证集上的的error都存进了这个error2的容器里 
	j=0;//j也是第一问里声明了的，这里清零一下还能用，不用重新声明一个新的变量
	for(int i=0;i<81;i++){
//		cout<<"error2["<<i<<"]="<<error2[i]<<endl;
		if(error2[i]<error2[0]){
			error2[0]=error2[i];
			j=i;//j的可能性取值是0~80，共81种可能 
		}
	}//至此，我们找到了error最小的那个pair的编号 
	double best_lr=((j%9)+1)*0.1;
    double best_mt=((j+1-(j+1)%9)/9+1)*0.1;
    cout<<"The BEST pair of learning rate and momentum term is: ("<<best_lr<<','<<best_mt<<").\n";
	cout<<"And the validation mean square error(MSE) is: "<<error2[0]<<endl;    
    //至此，完成了STAGE 2的任务！
    cout<<endl;   
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------Stage 2 complete-----------------------------"<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"IN CONCLUTION: THE BEST STRUCTURE OF THE NUERUAL NETWORK IS: (input, hidden, output)=("<<best_in<<','<<best_hd<<','<<"1)\n";
    cout<<"THE BEST PARAMETERS OF THE NUERUAL NETWORK IS: (learing rate, momentum term)=("<<best_lr<<','<<best_mt<<")\n";
    
    
    
    //接下去解决第三个问题，即用这个精挑细选出来的特定结构、特定参数的神经网络，来做一次预测 
    _tdata=TrDatArg(rawdata,best_in,1,0.7);
    _vdata=ValiDatArg(rawdata,best_in,1,0.7,0.15);//因为我们只以最后的20%的数据为testing set，所以之前的80%就都是垫脚石 
    BP bp3;
    bp3.SetNodes(best_in,best_hd,1);
	bp3.SetParas(best_lr,best_mt);
	bp3.GetTrainData(_tdata);
	bp3.InitNetWork();
	bp3.Train();
	bp3.GetValiData(_vdata);
	double testingError=bp3.GetValiAccu(); 
	cout<<"The EORROR of testing is: "<<testingError<<endl;
	cout<<endl;   
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------Stage 3 complete-----------------------------"<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;
      
    return 0;
} 

