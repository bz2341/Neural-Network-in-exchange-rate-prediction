/*ע�⣺
�����������ģ�����������Ը����ĵط�
1.��δ��ÿ�������ṹ������ʹ�����ؿ��巽������ֲ��Բ�ͬ�������ʼȨ�������� 
2.��������������������ÿһ������ṹ������ʹ�����ؿ���ģ�⣬��ʵҲ���Գ�������Щ��ͬ�ṹ������ĳ�ʼ���Ȩ�������󻥲���ͬ���ڱ����У��������������ͬ�� 
��һ��ʹ��ʱҪע��ĵط���ֻ�ܶ�ȡ260�����ݣ����û��Ǹ���Ϊ������Դ��Excel�������Ǹɸɾ����ľ�260�����ݣ�������ﲻҪ�У���������Ҫ���಻�ٸպ�260��*/
#include<iostream>
#include<fstream>//ʹ���ļ����������
#include<string>
#include<string.h>
#include<stdio.h>
#include<cstdlib>
#include<ctime>
#include<cmath> 
#include"bp.h"
using namespace std;

Vector<Data> TrDatArg(double* r,int in,int out,double TrPortion){//�������������˼��training data arrange 
                         	//�������ר�Ÿ����һ������ȫ�����ݻ��߶������ݴ�����ض��ṹ-����
							 //(3,3,1)-����������ֱ�ӽ��ܵ�������ʽ�������ɡ�һϵ�еĻ���Ƭ�Ρ� 
	int size=260;
	int num=(int)(size*TrPortion-in);//���num������������¼training set���е��������������Ǽ�¼
	                                //training set�н����еġ���С������Ԫ���ķ���                                 
    Vector<Data> _tdata;
    for(int i=0;i<num;i++){
    	Data t;
    	for(int j=i;j<in+i;j++){t.xxx.push_back(r[j]);}
	    for(int j=0;j<out;j++){t.yyy.push_back(r[i+in+j]);}
	    _tdata.push_back(t);
	}          
	return _tdata;
}//���ˣ�_tdata���� 

Vector<Data> ValiDatArg(double* r,int in,int out,double TrPortion,double ValiPortion){
	/*int size=0; 
	while(r[size]!=10000){                
		size++;
	}*/
	int size=260;
	int n=(int)(size*TrPortion);//��ǰn����training�õ����ݼ��������n����ʼ����validate�õ����� 
	int num=(int)(size*ValiPortion-in);//���Ǽ���validate�õġ���С������Ԫ���ķ��� 
	Vector<Data> _vdata;              
    for(int i=0;i<num;i++){
    	Data t;
    	for(int j=i;j<in+i;j++){t.xxx.push_back(r[n+j]);}
	    for(int j=0;j<out;j++){t.yyy.push_back(r[n+i+in+j]);}
	    _vdata.push_back(t);
	}          
	return _vdata;
}//���ˣ�_vdata���� 

int main(){
	ifstream is;
	is.open("c:\\file dock\\exchange\\ExchangeRateData2016.csv");
	if(!is.is_open()){       
	cerr<<"���ļ�ʧ�ܣ�\n"; 
	return -1;
    }
    double rawdata[260];
    for(int i=0;i<260;i++){
    	is>>rawdata[i];
	}
    is.close();//���ˣ��Ѿ���csv�ļ��е�����ԭʼ���ݶ��뵽��һ����rawdata��double�������� 
    
    cout<<"these are the rawdata:\n";
    for(int i=0;i<260;i++){
    	cout<<rawdata[i]<<endl;
	}

    
   //�����������ĸ��������򣡣�����������������������
   //�Ҳ���ɾ������ע�ͣ���ΪΪ�˼����������������bug�������Եķ�����֤���̡��������޵���������������  
     
        
    
//���������һ�����⣬������learining rate=0.3��momentum term=0.3��������ѡ�����ŵ�(input nodes, hidden nodes)pair
//��ʵ������һ���Դ�(3,3,1)��(10,10,1)��������64�ֽṹ�������磬����һ���Ĳ��Ϻ�һ��ǿ�ȵ�training�£��ĸ�ģ�������ɵ���ã� 
    double error1[64]; int i=0; 
	Vector<Data> _tdata, _vdata; //����������������������Ժ�����Ϳ���һֱ�� 
    
	for(int in=3;in<=10;in++){
    	for(int hd=3;hd<=10;hd++){
    		_tdata=TrDatArg(rawdata,in,1,0.7);
    		_vdata=ValiDatArg(rawdata,in,1,0.7,0.1);//���ˣ����κ�һ�ֽṹ�����¶�׼���������ݿɹ�����ֱ��ʹ�õ����ݣ�ѵ�����ݺ���֤���� 
    		BP bp1;//���bp1�����ڲ�ѭ���������ģ��������Ǹ��ֲ��������ѣ�һ�����ѭ�������ˣ���������ͻ����� 
		    bp1.SetNodes(in,hd,1);
			bp1.SetParas(0.3,0.3);
			bp1.GetTrainData(_tdata);
			bp1.InitNetWork();
			cout<<"For (inputNodes, hiddenNodes, outputNodes)=("<<in<<','<<hd<<",1) network: \n";
			bp1.Train();
			bp1.GetValiData(_vdata);
			cout<<endl;
			error1[i]=bp1.GetValiAccu(); 
			i++;//����error1[0]~erroe1[7]��8������(3,3,1)~(3,10,1)��˸������validation error 
		}//i��ȡֵ��0~63 
	}//���ˣ����ǰѴ�(3,3,1)��(10,10,1)��64�����綼����һ�飬���õ�����������ͬѧϰǿ���¸��Եõ���validationʱ�������������error1��������� 
    //Ȼ���������ҳ�error��С���Ƕ�pair��
    int j=0;
	for(int i=0;i<64;i++){
//		cout<<"error1["<<i<<"]="<<error1[i]<<endl;
		if(error1[i]<error1[0]){
			error1[0]=error1[i];
			j=i;
		}
	}//��������ܹ�ʹ��error1����С��һ����������error1[0]��λ�ò�����ԭ���Ľű걻������j��,������������ԭ�����j��Ӧ��pair��
	int best_hd=(j%8)+3;
	int best_in=(j+1-(j+1)%8)/8+3;
    cout<<"The BEST pair of input nodes and hidden layer nodes is: ("<<best_in<<','<<best_hd<<").\n";
	cout<<"And the validation mean square error(MSE) is: "<<error1[0]<<endl;    
    //���ˣ������STAGE 1������
 
    cout<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------Stage 1 complete-----------------------------"<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;


//����ȥ����ڶ������⣬���������������ز�Ľڵ���Ϊ��ѵ��Ժ�������ѡ������ŵ�(learing rate,momentun term)pair:
//�����Ƶأ���һ����(learing rate,momentun term)��(0.1,0.1)��(0.9,0.9)��81��pair������һ���Ĳ��Ϻ�һ��ǿ�ȵ�training�£��ĸ�ģ�������ɵ���ã�
    
	double error2[81]; i=0;//������1�Ľ�������������ı���i�Ծɿ����ã���Ҫ�˷�
    _tdata=TrDatArg(rawdata,best_in,1,0.7);//���ڣ�����Ľṹ��һֱ�̶��˵ģ��Ͷ���(best_in,best_hd,1)��������������Ľṹ���ˣ���������
    _vdata=ValiDatArg(rawdata,best_in,1,0.7,0.1);//���ܵġ�һϵ�л���Ƭ�Ρ�����̬Ҳ�Ͷ��ˣ����Բ�������һ����������ÿ�ζ���������ѵ�����Ϻ���֤���� 
	for(double lr=0.1;lr<=0.9;lr=lr+0.1){
		for(double mt=0.1;mt<=0.9;mt=mt+0.1){
    		BP bp2;//���bp2�������bp1һ�����Ұ����Ǵ���ɾֲ���������������ѭ����������һ��ȫ�ֱ�����Ȼ��ÿ��ѭ���������޸������������Ϊ��������Ҫ��һ��ʵ��
			       //��ֻ����һ��������ÿ��ʵ���Ժ�Ͱ��������ϴϴ���á�������������������������һ��ʵ���е�ÿһ��ʱ����ȥ��һ��һ���Ե����ף����ʵ������
				   //����������ӵ����������ĺô��ǱȽϿɿ���ÿ��ʵ�鿪ʼʱ��������ȫ�µġ��ɾ��ģ�������Ϊ�ϴ�ʵ��Ĳ���ûϴ�ɾ������¶����ʵ���Ӱ�� 
		    bp2.SetNodes(best_in,best_hd,1);
			bp2.SetParas(lr,mt);//�����������������仯�������pair��֮ǰ���ù̶���(0.3,0.3) 
			bp2.GetTrainData(_tdata);
			bp2.InitNetWork();
			cout<<"For (learning rate, momentun term)=("<<lr<<','<<mt<<") network: \n";
			bp2.Train();
			bp2.GetValiData(_vdata);
			cout<<endl;
			error2[i]=bp2.GetValiAccu(); 
			i++;
		}
	}//���ˣ�������ͬ��ѧϰǿ�ȱ�������������Ҫ����81��learning rate��momentum term��pair������ÿһ������ͬ��֤���ϵĵ�error����������error2�������� 
	j=0;//jҲ�ǵ�һ���������˵ģ���������һ�»����ã�������������һ���µı���
	for(int i=0;i<81;i++){
//		cout<<"error2["<<i<<"]="<<error2[i]<<endl;
		if(error2[i]<error2[0]){
			error2[0]=error2[i];
			j=i;//j�Ŀ�����ȡֵ��0~80����81�ֿ��� 
		}
	}//���ˣ������ҵ���error��С���Ǹ�pair�ı�� 
	double best_lr=((j%9)+1)*0.1;
    double best_mt=((j+1-(j+1)%9)/9+1)*0.1;
    cout<<"The BEST pair of learning rate and momentum term is: ("<<best_lr<<','<<best_mt<<").\n";
	cout<<"And the validation mean square error(MSE) is: "<<error2[0]<<endl;    
    //���ˣ������STAGE 2������
    cout<<endl;   
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"----------------------------Stage 2 complete-----------------------------"<<endl;
    cout<<"-------------------------------------------------------------------------"<<endl;
    cout<<"IN CONCLUTION: THE BEST STRUCTURE OF THE NUERUAL NETWORK IS: (input, hidden, output)=("<<best_in<<','<<best_hd<<','<<"1)\n";
    cout<<"THE BEST PARAMETERS OF THE NUERUAL NETWORK IS: (learing rate, momentum term)=("<<best_lr<<','<<best_mt<<")\n";
    
    
    
    //����ȥ������������⣬�����������ϸѡ�������ض��ṹ���ض������������磬����һ��Ԥ�� 
    _tdata=TrDatArg(rawdata,best_in,1,0.7);
    _vdata=ValiDatArg(rawdata,best_in,1,0.7,0.15);//��Ϊ����ֻ������20%������Ϊtesting set������֮ǰ��80%�Ͷ��ǵ��ʯ 
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

