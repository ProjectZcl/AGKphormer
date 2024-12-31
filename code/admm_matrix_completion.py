import numpy as np
import pandas as pd

def soft_thresholding(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.)

def admm_matrix_completion(T, rho, alpha, max_iter=2, tol=1e-4):

    n, m = T.shape
    X = np.random.rand(n, m)
    Z = X.copy()
    L = np.zeros((n, m))

    for iter_num in range(max_iter):
        # 更新Z
        Z = soft_thresholding(X + L, alpha / rho)

        # 更新X，这里使用核范数的近似，即矩阵奇异值的和
        U, s, V = np.linalg.svd(Z + L, full_matrices=False)
        s = np.maximum(s - alpha / rho, 0)
        X = np.dot(U, np.dot(np.diag(s), V))

        # 投影回[0,1]区间
        X = np.clip(X, 0, 1)
        # X = X / np.sum(X, axis=1, keepdims=True) # 除以每行的和
        row_sums = np.sqrt((X ** 2).sum(axis=1))
        X = X / np.outer(row_sums, np.ones_like(X[0]))  # L2归一化

        X[T == 1] = 1

        # 更新L
        L = L + Z - X

        # 检查收敛
        if np.linalg.norm(X - Z, 'fro') < tol:
            break
        print(iter_num, np.linalg.norm(X - Z, 'fro'))

    return X

def main():
    # 读取数据
    Adj = pd.read_csv('../data/association_matrix.csv', header=0)  # 2315*265
    Dis_simi = pd.read_csv('../data/diease_network_simi.csv', header=0)  # 265*265
    Meta_simi = pd.read_csv('../data/metabolite_ntework_simi.csv', header=0)  # 2315*2315

    # 构建目标矩阵T
    T = np.vstack([np.hstack([Meta_simi, Adj]), np.hstack([Adj.T, Dis_simi])])

    # 参数设置
    rho = 1.0
    alpha = 0.01

    # 运行ADMM算法
    A_pred = admm_matrix_completion(T, rho, alpha)
    df_A_completed = pd.DataFrame(A_pred)
    df_A_completed.to_csv('../data/association_matrix_completed_ad_T.csv', index=False)

    # 提取预测矩阵
    A_pred = A_pred[:2315, 2315:]

    # 保存预测矩阵
    df_A_completed = pd.DataFrame(A_pred)
    df_A_completed.to_csv('../data/association_matrix_completed_ad_T1.csv', index=False)

    print("预测矩阵已保存。")

if __name__ == "__main__":
    main()