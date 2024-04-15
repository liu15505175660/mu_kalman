#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>

//这里，我就不多放任何的头文件了，会将所有的部分进行回车分开
//如果想更简介，分开头文件进行，别忘了预处理声明
class kalmanfilter
{
    public:
    int dim_x;//这里，这是定义的状态向量的维度
    int dim_z;//观测向量的维度
    int dim_u;//控制输入向量的维度
    Eigen::VectorXd x_p_k;
    Eigen::VectorXd x_l_k;
    Eigen::VectorXd x_k;
    Eigen::VectorXd z_k;
    Eigen::MatrixXd A;
    Eigen::MatrixXd K;
    
    Eigen::MatrixXd P;
    Eigen::MatrixXd R;
    public:
    kalmanfilter(int dim_x_, int dim_z_, int dim_u_)//这里本来想利用私有传参，但发现下面继承过不去
    {
        dim_x = dim_x_;
        dim_z = dim_z_;
        dim_u = dim_u_;

        x_p_k = Eigen::VectorXd::Zero(dim_x);//VectorXd 类表示大小可变的向量,通常是列向量。第二个参数6就表示为列
        //因为我们是imu，6种状态参数，dim_x表行数
        x_l_k = Eigen::VectorXd::Zero(dim_x);
        x_k = Eigen::VectorXd::Zero(dim_x);
        z_k = Eigen::VectorXd::Zero(dim_z);

        A = Eigen::MatrixXd::Identity(dim_x,dim_x);//这步是创建单位矩阵的，大小为dim_x,dim_x
        K = Eigen::MatrixXd::Zero(dim_x,dim_z);//同理，大小为dim_x,dim_z的零矩阵
        Q = Eigen::MatrixXd::Zero(dim_x,dim_x);
        P = Eigen::MatrixXd::Zero(dim_x,dim_x);
        R = Eigen::MatrixXd::Zero(dim_z,dim_z);
    }
    Eigen::MatrixXd Q;
    //接下来是定义纯虚函数，进行K的初始化，预测与更新，因为卡尔曼核心就是K
    //不过这里更加详细的解释，我大概要复兴完多态才能写
    virtual void init(Eigen::VectorXd &x_k) = 0;

    virtual Eigen::VectorXd predict(Eigen::VectorXd &u,double t) = 0;

    virtual Eigen::VectorXd update(Eigen::VectorXd &z_k) = 0;

};



class kalmanFILTER : public kalmanfilter//继承，kalmanFILTER继承了kalmanfilter中的public与protect
//这意味着，kalmanFILTER会继承kalmanfilter中的public成员变量和函数，并可以通过指针与引用来进行访问
{
    public:
    Eigen::MatrixXd B;
    Eigen::MatrixXd H;

    kalmanFILTER(int dim_x, int dim_z, int dim_u):kalmanfilter(dim_x , dim_z, dim_u)
    //这里是因为派生类在构造时，需要显示调用基类的构造函数进行初始化
    //指定积累的构造函数来初始化基类的成员，才不会导致编译错误
    {
        if (dim_u > 0)//这里主要是利用输入的维度来确定B矩阵的列数
        //如果说并没有输入，那么B矩阵列数需要与状态向量的维数相同
        {
            B = Eigen::MatrixXd::Zero(dim_x,dim_u);
        }
        else
        {
            B = Eigen::MatrixXd::Zero(dim_x,dim_x);//实际中，如果没有输入，那么B矩阵常会设置为与状态向量维度相同的零矩阵
            H = Eigen::MatrixXd::Zero(dim_z,dim_x);
        }
    }
        //接下来的这段有些复杂，我不确定是否有更简单的方式可以完成这个操作
    kalmanFILTER(int dim_x, int dim_z, int dim_u, const Eigen::MatrixXd Q, const Eigen::MatrixXd R, const  Eigen::MatrixXd B)
    :kalmanFILTER(dim_x, dim_z, dim_u)
    //这里的也是一个基类的初始化，但其实我不能很好解释他，只能说以后再好好理解
    {
        this->Q = Q;
        this->R = R;
        this->B = B;
    }

    void init(Eigen::VectorXd &x_k) override;//override可以帮助我们在代码编写阶段就发现可能的错误

    Eigen::VectorXd predict(Eigen::VectorXd &u, double t) override;

    Eigen::VectorXd update(Eigen::VectorXd &z_k) override;

};


//总而言之，上面就是一些定义与初始化的过程

//那么接下来，我们进行五项黄金公式的书写

//这里是一个类内的声明，类外的函数实现，，之所以前面加kalmanFILTER，是因为我们要声明在该类中的init，使得更加清晰
//当然了为觉着不加也错不了
void kalmanFILTER::init(Eigen::VectorXd &x_k)
{
    //赋值，因为这里是初始化操作，所以无论是预测还是更新后，我们都将其给予当前值
    this->x_p_k = x_k;
    this->x_l_k = x_k;
    this->P = Eigen::MatrixXd::Zero(dim_x, dim_x);
    //对协方差矩阵的一个初始化
}

Eigen::VectorXd kalmanFILTER::predict(Eigen::VectorXd &u,double t)//与上述同理，这里，我们设置了时间间隔
{
    double delta_t = t;//设立时间间隔
    for (int i = 0; i < dim_x/2; i++)
    {
        A(i,dim_x/2+i) = delta_t;
    }
    //更新转移矩阵，以反映给定时间间隔内的变化

    x_p_k = A*x_l_k + B*u;//黄金一条
    P = A*P*A.transpose()+Q;//黄金五条

    return x_p_k;
}

Eigen::VectorXd kalmanFILTER::update(Eigen::VectorXd &z_k)
{
    for (int i = 0; i < dim_z; i++)
    {
        H(i,i)=1;//这个就是定义观测矩阵的中心线，1的意思就是说我们不对状态向量进行任何的线性变换或缩放
    }
    K = P*H.transpose()*(H*P*H.transpose()+R).inverse();//黄金2条
    x_k =x_p_k +K*(z_k-H*x_p_k);//黄金3条
    P=P-K*H*P;//黄金4条
    x_l_k = x_k;
    
    return x_k;
}


int main()
{
    std::string input_filename = "input.csv";
    std::string output_filename = "output.csv";

    std::ifstream input_file(input_filename);
    std::ofstream output_file(output_filename);

    if (!input_file.is_open() || !output_file.is_open()) 
    {
        std::cerr << "Error opening files!" << std::endl;
        return -1;
    }



    //初始化卡尔曼滤波
    int dim_x = 6;
    int dim_z = 6;
    int dim_u = 6;

// 定义矩阵和向量
    Eigen::MatrixXd Q(6, 6);
    Q << 0.1, 0, 0, 0, 0, 0,
         0, 0.1, 0, 0, 0, 0,
         0, 0, 0.1, 0, 0, 0,
         0, 0, 0, 0.1, 0, 0,
         0, 0, 0, 0, 0.1, 0,
         0, 0, 0, 0, 0, 0.1;
    Eigen::MatrixXd R(6, 6);
    R << 0.1, 0, 0, 0, 0, 0,
         0, 0.1, 0, 0, 0, 0,
         0, 0, 0.1, 0, 0, 0,
         0, 0, 0, 0.1, 0, 0,
         0, 0, 0, 0, 0.1, 0,
         0, 0, 0, 0, 0, 0.1;
         
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6, 1);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(1);



    // 初始化卡尔曼滤波器
    auto kf = std::make_shared<kalmanFILTER>(6, 6, 0, Q, R, B);
    Eigen::VectorXd x_0 = Eigen::VectorXd::Zero(6);
    kf->init(x_0);

     std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream ss(line);
        Eigen::VectorXd z_k(6);
        double value;
        int index = 0;
        while (ss >> value) {
            if (index < 6) {  // 假设CSV只有6个测量值
                z_k(index++) = value;
            }
            if (ss.peek() == ',') ss.ignore();
        }

        Eigen::VectorXd x_p_k = kf->predict(u, 1.0);  // 假设时间步长为1秒
        Eigen::VectorXd x_k = kf->update(z_k);

        // 写入过滤后的数据到输出文件
        for (int i = 0; i < x_k.size(); ++i) {
            output_file << x_k(i);
            if (i < x_k.size() - 1) output_file << "\t";
        }
        output_file << std::endl;
    }

    input_file.close();
    output_file.close();

    std::cout << "Data filtering complete. Results saved to " << output_filename << std::endl;

    return 0;
}