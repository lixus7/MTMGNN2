# MetroPrediction


************************************************************************************************************************
GAT应用到STGCN里面，加残差效果才收敛，

残差 + tanh 激活： MSE, RMSE, MAE, MAPE, 1370.4246210426, 37.0192466299, 20.9699514134, 22.6429510010


残差+ embed_dim （ 3）  + no activate  + zscore    2     MSE, RMSE, MAE, MAPE, 1399.7646479528, 37.4134287115, 20.9906162721, 22.9146834089

残差+ embed_dim （ 3，hidden ）  + no activate  + zscore      3     MSE, RMSE, MAE, MAPE, 1365.4638823748, 36.9521837294, 21.0019744013, 22.5974337964

残差+ embed_dim （ 3，hidden ）  + tanh + zscore   3         MSE, RMSE, MAE, MAPE, 1364.3915199044, 36.9376707428, 21.0065881190, 22.5718686191

残差+ embed_dim （ 64 ）  + no activate  + zscore      4     MSE, RMSE, MAE, MAPE, 1357.1699547268, 36.8397876585, 20.9673940804, 22.3889753685

残差+ embed_dim （ 64 ）  + tanh  + zscore    4      MSE, RMSE, MAE, MAPE, 1329.5515385401, 36.4630160374, 20.6563147774, 22.1648953793
    
残差   + embed_dim （ 3 ）  + tanh + zscore    cuda  2    1363     36.92    20.99   23.42

残差   + embed_dim （ 3 ）  + tanh + min max      5     MSE, RMSE, MAE, MAPE, 1270.0372356572, 35.6375817875, 20.6922131502, 22.9604903100         


残差   + embed_dim （ 64 ）  + tanh  +     min max      6        MSE, RMSE, MAE, MAPE, 1309.6744373076, 36.1894243849, 20.8942361457, 23.9292927351

残差+ embed_dim （ 3，hidden ）  +  tanh  +     min max     7    无收敛     MSE, RMSE, MAE, MAPE, 1328.4361467771, 36.4477179914, 20.8662970323, 24.0861296614


STGCN   GCN + GAT    embed_dim （ 3 ）  + tanh + min max       cuda 0    MSE, RMSE, MAE, MAPE, 1286.1435756541, 35.8628439426, 20.7329524583, 22.8905257771

STGCN   GCN + GAT    embed_dim （ 3 ）  + tanh + min max        cuda 4            1290  35.92，20.75，23.32 

STGCN   GCN + GAT +x    embed_dim （ 3 ）  + tanh + min max    残差        cuda 5       1269    35.63   20.59   22.88



Transformer  1791.97  42.332  24.498  31.764

GAT1_Trans  + embed_dim （ 3，hidden ）  + tanh + zscore   (64 + 8)         MSE, RMSE, MAE, MAPE, 2251.637, 47.451, 26.044, 37.862   

GAT1_Trans  残差  + embed_dim （ 3 ）+tanh + zscore   ( 32 + 4 ) cuda 1      MSE, RMSE, MAE, MAPE, 1830.637, 42.786, 24.608, 34.234


GAT1_Trans  残差  + embed_dim （ 3 ） GAT+x  _cat x  +tanh + zscore   ( 32 + 4 )    177 cuda 1    MSE, RMSE, MAE, MAPE, 1771.956, 42.095, 23.955, 31.530

GAT2_Trans  残差  + embed_dim （ 3 ）+tanh + zscore   ( 32 + 4 )    177 cuda 2                              MSE, RMSE, MAE, MAPE, 1833.716, 42.822, 24.669, 32.707

GAT2_Trans  残差  + embed_dim （ 3 ） GAT  _cat x  +tanh + zscore   ( 32 + 4 )     177 cuda 3         MSE, RMSE, MAE, MAPE, 1716.708, 41.433, 23.763, 32.999

GAT1_Trans  残差  + embed_dim （ 3 ） GAT  _cat x  +tanh + zscore   ( 32 + 4 )    177 cuda 0（4）    MSE, RMSE, MAE, MAPE, 1695.057, 41.171, 23.683, 30.848

GAT + DCRNN   cuda 3

************************************************************************************************************************
************************************************************************************************************************
************************************************************************************************************************

************************************************************************************************************************
************************************************************************************************************************
************************************************************************************************************************
Transformer

embed  64   head  8    forward_expansion  4    MSE     MSE, RMSE, MAE, MAPE, 1915.002, 43.761, 25.606, 37.860

embed  64   head  8    forward_expansion  4    MSE  no relu    MSE, RMSE, MAE, MAPE, 2070.432, 45.502, 26.445, 40.686

embed  64   head  8    forward_expansion  4    MAE  no relu     MSE, RMSE, MAE, MAPE, 2037.709, 45.141, 25.564, 34.345

no relu  会变差

embed  64   head  8    forward_expansion  4    MAE  tanh  0   MSE, RMSE, MAE, MAPE, 2166.702, 46.548, 25.674, 33.509


embed  64   head  8    forward_expansion  4  MAE  MSE, RMSE, MAE, MAPE, 1992.193, 44.634, 25.101, 32.581

embed  32   head  8    forward_expansion  4   MAE     MSE, RMSE, MAE, MAPE, 1819.783, 42.659, 24.524, 32.275

embed  8   head  8    forward_expansion  4  MAE   MSE, RMSE, MAE, MAPE, 2209.889, 47.009, 25.789, 34.435

以下loss 全部为MAE

embed  32   head  4    forward_expansion  4  drop0.1    5      MSE, RMSE, MAE, MAPE, 2493.527, 49.935, 26.800, 34.075

不能drop

embed  32   head  4    forward_expansion  4                    MSE, RMSE, MAE, MAPE, 1791.970, 42.332, 24.498, 31.764           best    MSE, RMSE, MAE, MAPE, 1779.060, 42.179, 24.372, 32.415

embed  32   head  4    forward_expansion  4     ELU        MSE, RMSE, MAE, MAPE, 1950.273, 44.162, 25.074, 32.304

embed  32   head  4    forward_expansion  4  minmax     MSE, RMSE, MAE, MAPE, 2167.774, 46.559, 25.667, 28.269

embed  32   head  4    forward_expansion  4     feed_forward=linear   MSE, RMSE, MAE, MAPE, 1957.737, 44.246, 25.303, 35.105

embed  32   head  4    forward_expansion  4    Po1   177 cuda2     MSE, RMSE, MAE, MAPE, 1855.970, 43.081, 24.879, 32.244



embed  32   head  4    forward_expansion  2     0      MSE, RMSE, MAE, MAPE, 1879.726, 43.356, 24.700, 31.931

embed  32   head  4    forward_expansion  1     4     MSE, RMSE, MAE, MAPE, 1942.667, 44.076, 25.179, 33.533

embed  32   head  1   forward_expansion  4     1     MSE, RMSE, MAE, MAPE, 1843.055, 42.931, 24.495, 31.732

embed  32   head  2    forward_expansion  4     0     MSE, RMSE, MAE, MAPE, 1792.972, 42.344, 24.475, 32.238

embed  32   head  8    forward_expansion  4    2  去掉schedular    MSE, RMSE, MAE, MAPE, 2399.492, 48.985, 26.966, 39.975

不能去掉sche 

embed  512   head  8    forward_expansion  4       MAE relu  0       MSE, RMSE, MAE, MAPE, 2865.193, 53.527, 28.064, 37.438

embed  64   head  8    forward_expansion  4        layer 1          1     MSE, RMSE, MAE, MAPE, 2000.883, 44.731, 25.394, 35.572

embed  64   head  8    forward_expansion  4        layer 2          2      MSE, RMSE, MAE, MAPE, 2249.836, 47.432, 26.018, 35.471
************************************************************************************************************************
************************************************************************************************************************
************************************************************************************************************************
************************************************************************************************************************
model name : STTN  

github url :  https://github.com/wubin5/STTN

paper : (挂载Arxiv投稿中，估计难中。。。。)

Xu, M., Dai, W., Liu, C., Gao, X., Lin, W., Qi, G. J., & Xiong, H. (2020). Spatial-temporal transformer networks for traffic flow forecasting. arXiv preprint arXiv:2001.02908.

************************************************************************************************************************

model name : ST-GRAT       ( 参考Transformer的架构写的 ，pytorch架构

github url :  https://github.com/LMissher/ST-GRAT

paper : (CIKM 2020)

Park, C., Lee, C., Bahng, H., Tae, Y., Jin, S., Kim, K., ... & Choo, J. (2020, October). ST-GRAT: A novel spatio-temporal graph attention networks for accurately forecasting dynamically changing road speed. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 1215-1224).
************************************************************************************************************************
model name : ST-CGA       (  keras架构

github url : https://github.com/jillbetty001/ST-CGA

paper : (CIKM 2020)

Zhang, X., Huang, C., Xu, Y., & Xia, L. (2020, October). Spatial-temporal convolutional graph attention networks for citywide traffic flow forecasting. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 1853-1862).

************************************************************************************************************************
model name : ST-MGAT      ( dgl库，pytorch架构

github url : https://github.com/Kelang-Tian/ST-MGAT

paper : (ICTAI 2020)

Tian, K., Guo, J., Ye, K., & Xu, C. Z. (2020, November). ST-MGAT: Spatial-Temporal Multi-Head Graph Attention Networks for Traffic Forecasting. In 2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI) (pp. 714-721). IEEE.

************************************************************************************************************************

************************************************************************************************************************

其他task GAT：

************************************************************************************************************************

model: MTAD-GAT    (时间序列异常检测，pytorch架构

github url :  https://github.com/ML4ITS/mtad-gat-pytorch/blob/main/mtad_gat.py

paper :  (ICDM 2020)

Zhao, H., Wang, Y., Duan, J., Huang, C., Cao, D., Tong, Y., ... & Zhang, Q. (2020, November). Multivariate time-series anomaly detection via graph attention network. In 2020 IEEE International Conference on Data Mining (ICDM) (pp. 841-850). IEEE.

************************************************************************************************************************

model: 
