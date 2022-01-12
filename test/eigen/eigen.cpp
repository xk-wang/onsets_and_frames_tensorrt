// 学习教程 https://blog.csdn.net/hongge_smile/article/details/107296658
// 官方文档 https://eigen.tuxfamily.org/dox/
// 使用MAP类将eigen矩阵和数组进行类型转换 https://blog.csdn.net/u011521131/article/details/77862269
// 高维矩阵 https://blog.csdn.net/u013241583/article/details/109359123
// 高维矩阵堆叠 https://blog.csdn.net/luffytom/article/details/105460918

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

int main(){
    Eigen::MatrixXd m(2,2);
    m(0, 0)=3, m(1, 0)=2.5, m(0, 1)=-1, m(1, 1)=m(1, 0)+m(0, 1);
    std::cout << m << std::endl;

    Eigen::MatrixXd n = Eigen::MatrixXd::Random(3, 3);
    n = (n+Eigen::MatrixXd::Constant(3, 3, 1.2))*50;
    std::cout << "n = " << std::endl << n << std::endl;
    Eigen::VectorXd v(3);
    v << 1, 2, 3;
    std::cout << "n *v = " <<std::endl << n*v << std::endl;

    Eigen::RowVectorXd vec1(3);
    vec1 << 1, 2, 3;
    std::cout << "vec1 = " << vec1 << std::endl;
    Eigen::RowVectorXd vec2(4);
    vec2 << 1, 4, 9, 16;
    std::cout << "vec2 = " << vec2 << std::endl;
    Eigen::RowVectorXd joined(7);
    joined << vec1, vec2;
    std::cout << "joined = " << joined << std::endl;

    Eigen::Matrix4d m4 = Eigen::Matrix4d::Identity();
    std::cout << "m4 = " << std::endl << m4 << std::endl;

    std::cout << m4.size() << " " << m4.rows() << " " << m4.cols() << std::endl;
    
    Eigen::MatrixXd m5(4, 4);
    m5.resize(5, 4);
    // std::cout << "m4 = " << std::endl << m4 << std::endl;

    Eigen::ArrayXXf a(3, 3);
    Eigen::ArrayXXf b(3, 3);
    a << 1, 2, 3,
         4, 5, 6,
         7, 8,  9;
    b << 1, 2, 3,
         1, 2, 3,
         1, 2, 3;
    std::cout << "a+b = " << std::endl << a+b << std::endl;
    std::cout << "a-b = " << std::endl << a-b << std::endl;
    std::cout << "a*b = " << std::endl << a*b << std::endl;
    std::cout << "a/b = " << std::endl << a/b << std::endl;
    Eigen::ArrayXXf c= a-b; // << a-b;
    // c << a-b;
    // 比较每个元素位置中选择元素值最小的那个元素
    std::cout << c.min(a) << std::endl;

    // .matrix和.array必须要显示进行，array弥补了一些数学中没有定义的一些矩阵运算
    std::cout << "c = " << std::endl << c << std::endl;
    // auto d = c.matrix();
    c.transposeInPlace();
    std::cout << "c = " << std::endl << c << std::endl;


    // 点积和叉积 dot cross cross只可以用于Vector3向量
    Eigen::Vector3d x(1, 2, 3);
    Eigen::Vector3d y(0, 1, 2);
    std::cout << "vector product: " << x.adjoint()*y << std::endl;
    std::cout << "Dot product: " << x.dot(y) << std::endl;
    std::cout << "Cross product: " << std::endl << x.cross(y) << std::endl;
    
    // 基本的一些矩阵运算
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;
    std::cout << "the sum: " << mat.sum() << std::endl;
    std::cout << "the prod: " << mat.prod() << std::endl;
    std::cout << "the mean: " << mat.mean() << std::endl;
    std::cout << "the minCoeff: " << mat.minCoeff() << std::endl;
    std::cout << "the maxCoeff: " << mat.maxCoeff() << std::endl;
    std::cout << "the trace: " << mat.trace() << std::endl;

    Eigen::Vector2d mm = mat.diagonal();
    std::cout << "diagonal: " << std::endl << mm.sum() << std::endl;

    // 特殊的块操作 行和列
    Eigen::MatrixXf xx(4 ,4);
    for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
            xx(i, j) = j+1+i*4;
        }
    }
    std::cout << "xx = " << std::endl << xx << std::endl;
    std::cout << "2nd row: " << xx.row(1) << std::endl;
    xx.col(2) += 3*xx.col(0);
    std::cout << "xx = " << std::endl << xx << std::endl; 

    // 边角计算
    Eigen::Matrix4f mn;
    mn << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16;
    std::cout << "left 2 cols: " << std::endl << mn.leftCols(2) << std::endl;

    // 范数计算 也非常常用

    // 布尔规约 包含all any 和 count 含义同matlab
    Eigen::ArrayXXf bb(2, 2);
    bb << 1, 2,
         3, 4;
    std::cout << (bb>0).all() << std::endl;
    std::cout << (bb>0).any() << std::endl;
    std::cout << (bb>0).count() << std::endl;
    std::cout << (bb>2).all() << std::endl;
    std::cout << (bb>2).any() << std::endl;
    std::cout << (bb>2).count() << std::endl;


    // 获取元素的位置
    Eigen::MatrixXf::Index minRow, minCol;
    float min = bb.minCoeff(&minRow, &minCol);
    std::cout << "the min val is: " << min << " at: " << minRow << ", " << minCol << std::endl;

    // 部分规约 按照行或者列来进行操作
    Eigen::MatrixXf mmat(2, 4);
    mmat << 1, 2, 6, 9,
            3, 1, 7, 2;
    std::cout << "column's maximum: " << std::endl
              << mmat.colwise().maxCoeff() << std::endl;
    std::cout << "row's maximum: " << std::endl
              << mmat.rowwise().maxCoeff() << std::endl;

    // 广播机制 类似部分规约
    Eigen::MatrixXf mnat(2, 4);
    Eigen::VectorXf vv(2);
    mnat << 1, 2, 6, 9,
           3, 1, 7, 2;
    vv << 0,
          1;
    mnat.colwise() += vv; // 必须要明确指定广播的方向，无法进行自动推断
    std::cout << "the mat + vector = " << std::endl << mnat << std::endl;

    // Geometry 几何模块 包含旋转和一些其他的参数

    // 稠密问题的叙述矩阵分解

    // 将cpp二维矩阵和eigen矩阵进行转换
    std::vector<std::vector<float>> xy;
    for(int i=0;i<3;++i){
        std::vector<float>aaa;
        for(int j=0;j<3;++j){
            aaa.push_back(j);
        }
        xy.push_back(aaa);
    }
    Eigen::MatrixXf re(3, 3);
    for(int i=0;i<3;++i){
        re.row(i) = Eigen::VectorXf::Map(&xy[i][0], xy[i].size());
    }
    std::cout << "res = " << std::endl << re << std::endl;
    
    return 0;
}