#include "mannwhitneyu.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace hpdexc;
using namespace hpdexc::tensor;

int main() {
    std::cout << "开始测试 mannwhitneyu 函数..." << std::endl;
    
    try {
        // 创建测试数据
        const std::size_t R = 4;  // 行数
        const std::size_t C = 2;  // 列数
        const std::size_t n_targets = 1;  // 目标组数
        
        std::cout << "创建测试数据: R=" << R << ", C=" << C << ", n_targets=" << n_targets << std::endl;
        
        // 创建稀疏矩阵数据（每列每组≥2个样本）
        // 列0: 行0,1,2,3；列1: 行0,1,2,3
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0,
                                    5.0, 6.0, 7.0, 8.0};
        std::vector<int32_t> indices = {0, 1, 2, 3,
                                         0, 1, 2, 3};
        std::vector<int32_t> indptr = {0, 4, 8};
        
        // 创建CSC矩阵
        auto csc = Csc<double, int32_t>(
            data.data(),
            indptr.data(),
            indices.data(),
            R, C, data.size()
        );
        
        std::cout << "CSC矩阵创建成功: " << csc.rows() << "x" << csc.cols() << std::endl;
        
        // 创建group_id（两组：每组各2行）
        std::vector<int32_t> group_id = {0, 1, 0, 1};
        auto group_vec = Vector<int32_t>(group_id.data(), group_id.size());
        
        std::cout << "group_id创建成功: ";
        for (size_t i = 0; i < group_id.size(); ++i) {
            std::cout << group_id[i] << " ";
        }
        std::cout << std::endl;
        
        // 创建选项
        MannWhitneyuOption<double> opt;
        opt.ref_sorted = false;
        opt.tar_sorted = false;
        opt.tie_correction = true;
        opt.use_continuity = true;
        opt.sparse_type = MannWhitneyuOption<double>::SparseValueMinmax::none;
        opt.sparse_value = 0.0;
        opt.alternative = MannWhitneyuOption<double>::Alternative::two_sided;
        opt.method = MannWhitneyuOption<double>::Method::exact;
        
        std::cout << "选项创建成功" << std::endl;
        
        // 调用mannwhitneyu函数
        std::cout << "调用 mannwhitneyu 函数..." << std::endl;
        
        auto result = mannwhitneyu<double, int32_t>(
            csc,
            opt,
            group_vec,
            n_targets,
            1,  // threads
            nullptr  // progress_ptr
        );
        
        std::cout << "✓ mannwhitneyu 函数调用成功!" << std::endl;
        
        auto [U_mat, P_mat] = result;
        std::cout << "U矩阵形状: " << U_mat.shape()[0] << "x" << U_mat.shape()[1] << std::endl;
        std::cout << "P矩阵形状: " << P_mat.shape()[0] << "x" << P_mat.shape()[1] << std::endl;
        
        // 打印结果
        if (U_mat.data() != nullptr && U_mat.size() > 0) {
            std::cout << "U矩阵数据: ";
            for (size_t i = 0; i < U_mat.size(); ++i) {
                std::cout << U_mat.data()[i] << " ";
            }
            std::cout << std::endl;
        }
        
        if (P_mat.data() != nullptr && P_mat.size() > 0) {
            std::cout << "P矩阵数据: ";
            for (size_t i = 0; i < P_mat.size(); ++i) {
                std::cout << P_mat.data()[i] << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ 错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "✗ 未知错误" << std::endl;
        return 1;
    }
    
    std::cout << "测试完成!" << std::endl;
    return 0;
}
