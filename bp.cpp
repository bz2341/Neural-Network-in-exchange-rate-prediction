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
//����������������û���������ʽ�ض��뵽��д�õ��������������� 
//Ҳ����˵�ڷ�����֮�⣬��main�����У��Ѿ���һ������ "_data"��vector�����ˣ�������װ��һϵ�еġ���С������Ԫ�� 
//Ȼ����������Ĺ����ǽ���һϵ�е�����һ����Ԫ��ֵ��BP������ж����һ����data��Vector<Data>�����ܣ���ʵ����
//��data�������������������_data����������ͬ���������� 

void BP::GetValiData(const Vector<Data> _vdata){
	vdata=_vdata;
}

void BP::SetNodes(int in,int hd,int ou){//���BP��Ĺ��г�Ա�����������ǣ��ѷ������������Ľڵ������뵽 
	in_num=in;                              //���������У��Թ�֮���ڷ�����������һ��������������� 
	hd_num=hd;
	ou_num=ou;
}

void BP::SetParas(double lr,double mt){
	Lerate=lr;
	Moterm=mt; 
}

void BP::InitNetWork(){//�����Ա�����������ǣ�����������ľ��������ɳ�ʼ��Ϊ�������Ȩ�������� 
	srand(time(NULL));
    memset(w,0,sizeof(w));//�Ȱ���������װȨ�ؾ��������W����ʼ��Ϊ0,Ȼ�������ֵһ������� 
	memset(Lw,0,sizeof(Lw));//�����W���һ�������� 
	for(int i=0;i<in_num;i++){
		for(int j=0;j<hd_num;j++){
		   
			w[0][i][j]=1.0*rand()/RAND_MAX-0.5;//�����ڲ���(-0.5,0.5)����������������������������������ز�֮���Ȩ�ؾ��� 
            Lw[0][i][j]=w[0][i][j];	
            
		} 
	} //������һ��Ҫע��ģ�W�ĵ�һ��ά�Ⱦ���layer������C++������ű��Ǵ�0��ʼ�ģ����Ե�һά��0�ͱ�ʾ��һ��layer���� 
    for(int i=0;i<hd_num;i++){//���������ز�֮����Ǹ�Ȩ�ؾ���layer������һά��1�ͱ�ʾ�ڶ��㣬�����ز�������֮��� 
    	for(int j=0;j<ou_num;j++){//�ǲ�Ȩֵ���� 
		    
    	    w[1][i][j]=1.0*rand()/RAND_MAX-0.5;//������(-0.5,0.5)���������䣨��ʼ�������ز�������֮���Ȩ�ؾ��� 
		    Lw[1][i][j]=w[1][i][j];//�ڳ�ʼ����ʱ���w��Lw����������ȡ����ȫһ���ġ��������������ǣ��ڵ�һ��ѧϰ֮ǰ��	
		
		}//����λ0 
	}
}

void BP::Train(){//ǧ���ס����ѵ������֮ǰҪ����������ľ���ṹ����ʼ�����磡���� 
	cout<<"Begin to train BP Neural Network!\n"; 
	int num=tdata.size(); //֮ǰ�Ѿ�˵����data��������һϵ�еġ���С������Ԫ��������������Vector�����
	                      //Vector�����Դ���һ��size()�����ġ���ô���data������size�����⡰һϵ����С������Ԫ��
						  //�ﵽ���ж��ٷ����֡���С������Ԫ���������ڣ�3,3,1��������ṹ�£���70%*260=182������
						  //����ѧϰ������182�����������ɡ�3������ֵ��1�����ֵ���ġ���С������Ԫ���Ļ�����
						  //����179�ݣ���ô�������������£����data������size����179����Ϊ������179�ݡ���С������Ԫ�� 
	
    for(int iter=0;iter<ITERS;iter++){//��һ��ѭ����ѭ��һ�ξ������������training�����ǲ���3000��epoch 
    	for(int cnt=0;cnt<num;cnt++){//��һ��ѭ����ѭ��һ�ξ��ǲ���һ��epoch��Ҳ���ǰ�179�ݡ���С������Ԫ������һ��
		                  //��ʵ�ϣ����е�num����data��size�������ǵ�������num=179��cnt����0������178���ܹ�Ҳ��179�� 
    		for(int i=0;i<in_num;i++){x[0][i]=tdata.at(cnt).xxx[i];}//data.at(cnt)��ʾȡdata���Vector�����ĵ�cntά�� 
    		/*���at()������Vector���Լ������һ�����к�������ȡdata�������������ĵ�cnt�ݡ���С������Ԫ����
			����ȡ�����Ķ�������һ��Data���͵Ľṹ��Ȼ��ȡ���Data�����е�x��������ȡ���x�����ĵ�i��ֵ*/ 
		    ForwardTransfer();  //˳����һ�� 
			ReverseTransfer(cnt); //������һ�Σ�����һ��Ȩ������ 
		}  //���ˣ�һ��epoch�������  
	} //���ˣ����ڸ����ṹ�������磬���磨3,3,1����3000��epoch�Ѿ�������������ض��ṹ��������ԣ�ѧϰ���� 
//	double TrainingFinalError=GetError(num-1); //��3000��epoch�����һ��epoch�����һСƬ�����ѧϰ�������Ϊ����������ѧϰ
	                                 //���̣���179*3000�������������� 
 	cout<<"Training has been completed!\n";
//<<" The final trianing squared error of this "<<ITERS;
//	cout<<" epoches is: "<<GetError(num-1)<<endl;
//	cout<<"Please NOTE that this is just the final error after training set.\n";
//	cout<<"We haven't applied this training result on the validation set yet.\n";
} 

void BP::ForwardTransfer(){
	for(int j=0;j<hd_num;j++){
		Type s1=0;
		for(int i=0;i<in_num;i++){s1+=w[0][i][j]*x[0][i];}//s��ʾ���ز�ÿ���ڵ������ֵ���ڱ��������ǲ����Ƿ�ֵ��ƫ��ֵ
		                                                 //ÿ��x[0][i]��ֵ����Train()�����ڵ��ñ�����֮ǰ�ͻ������úõ� 
	x[1][j]=Sigmoid(s1);//��������ز��ÿһ���ڵ�����ֵ�� 
	}
	
	for(int j=0;j<ou_num;j++){//��ʵ�����ǵ������У�ou_num�����1,������д�������Ĺ�����ʽ����������벻���������ڻ��� 
	    Type s2=0;            //��ȥ��ѧϰ��δ����һ��Ԥ�⣬�����Զ�δ�����ಽԤ�� 
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

Type BP::GetValiAccu(){//��������������ǽ���training set��ѵ���õ�����Ӧ����һ��validation set���� 
	Type ans=0;        //validation set�ϵ�ÿһ�ݡ���С������Ԫ�����η�����������ǰ��һ�飬�õ������ 
	int num=vdata.size();//�������ƽ���ͣ������ƽ�������ƽ���� 
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

Vector<Type> BP::ForeCast(const Vector<Type> fdata){//ע�⣬������β�����Ҳ��data��������ֻ��һ����ʱ�Եĳƺ� 
                                                   //���ѣ���������Train()������ĳ�Ա����data�������¶�,����
												   //��������dataֻ��һ��double���͵�vector����Train()�������
												   //��Ա����data����һ��Data���͵�vector 
	int n=fdata.size();
    assert(n==in_num);// ����Ԥ��״̬����򵥵ķ�ʽ��ͨ����׼C���assert�ꡣ�����Ĳ�����һ������
	                  //���ʽ�������ʽ��ֵΪ�ٵ�ʱ��assert�����Դ�ļ��������������ͱ��ʽ���������ݣ�
					  //Ȼ���³����˳���Assert������ڴ��������ڲ���Ҫһ���Լ��ĳ���
	    //�ɼ���ForeCast()�����������ܵĲ�����ֻ��һ�����͡�һ����С������Ԫ��һ���ĵ������򣬲��ܽ���������
		//����Ƭ����Ϊ���� 
	for(int i=0;i<in_num;i++){x[0][i]=fdata[i];}
	ForwardTransfer();
	Vector<Type> v;
	for(int i=0;i<ou_num;i++){v.push_back(x[2][i]);}//����һ�������õ���push_back()����,c.push_back(X)����˼���ǰ�Ԫ��X���� 
	return v;	 //������c�����һλ��������������һ��vector��������ֻ����push_back������û���õȺŸ�ֵ 
}//�������������Ԥ��ģ����ǣ���ֻ�ǹ�����򵥵ġ�������Ԥ�⣬���磨3,3,1���������� ����ô���Ԥ�⺯�����ǽ���
 //һ��3ά��double������Ϊ������Ȼ�����һ��1άdouble������Ҳ����һ��doubleֵ��������Ϊ�����
 //�����������ƺ���� ForeCast����ֻ����һ��Ԥ�⣬ ����ʵҲδ�أ���Ϊ�ಽԤ����ʵ������ʵ�ַ�������һ��ÿ��ForwardTransfer
 //ֻ��ǰԤ��һ����Ȼ��һ��ForeCast��ForwardTransfer��Σ��ڶ���ÿ��ForwardTransfer��ǰԤ��ಽ��Ȼ��
 //һ��ForeCast��ForwardTransferһ�Ρ���Ȼ�������ַ���Ҳ�����ǵȼ۵ģ���Ϊ���õ�����Ϣ����һ����
 //anyway���ڱ����У���������ֻ��ForwardTransfer��ǰ��һ��Ԥ�⣬�����ForeCast�������ֽ���ʹ����ForwardTransfer���������
 //ForeCast�����ڱ�����ʵ���ô���ʵ�����񱾶εڶ���˵�����������Ԥ���һ��doubleֵ��δ��ֵ����������ʵ�����ó���Ҳ����
 //������һ�ж�������֮����stage3��ȥԤ��һ����һ���������ֵ�� 

void BP::ReverseTransfer(int cnt){
	CalcDelta(cnt);
	UpdateNetWork();
}

void BP::CalcDelta(int cnt){
	for(int j=0;j<ou_num;j++){
		d[1][j]=(x[2][j]-tdata.at(cnt).yyy[j])*x[2][j]*(A-x[2][j])/(A*B);
	} //����d�ĵ�һά��W�ĵ�һά�Ƕ�Ӧ�� 
	for(int i=0;i<hd_num;i++){
		Type t=0;
		for(int j=0;j<ou_num;j++){
			t+=d[1][j]*w[1][i][j];
		}
		d[0][i]=t*x[1][i]*(A-x[1][i])/(A*B);
	} 
}

void BP::UpdateNetWork(){
	//���ز�������֮���Ȩ�ص�����
	Type temp; 
	for(int i=0;i<hd_num;i++){
		for(int j=0;j<ou_num;j++){	
		    temp=w[1][i][j];
			w[1][i][j]=w[1][i][j]-Lerate*d[1][j]*x[1][i]+Moterm*(w[1][i][j]-Lw[1][i][j]);
		    Lw[1][i][j]=temp;//��temp������Lw,ʹ��ʼ����w����һ������ 
		}
	}
	//���������ز�ֱ�ӵ�Ȩ�ص�����
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
}//�ڱ����У����ǲ��������S�ͼ���� 

//���ˣ�����ʵ����ͷ�ļ��е����к��� 
