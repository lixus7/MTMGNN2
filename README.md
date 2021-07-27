# Metro

Default ---  minmax normalization
Baseline：                                            RMSE        MSE      MAE       MAPE
STGCN    -   		                        1383.155    37.191   21.296    23.045 

GATv1+TCN(stgcn                 best  1270.037    35.637     20.692    22.96                     Training time    60s/epoch
		(z-score)                 best  1329.551    36.463     20.656    22.16                     Training time    60s/epoch
1.1 GAT_DGL+ TCN(stgcn           best  1250.252    35.358     20.587    23.09                      Training time    10s/epoch
                      (z-score)                best  1282.515    35.812     20.649    22.81                      Training time    10s/epoch                  
GCN+GAT+ TCN(stgcn          best  1269          35.63       20.59      22.88                      Training time    88s/epoch 
		(z-score)                 best                                                                               Training time    88s/epoch 
2.2 GCN+GAT_DGl+ TCN(stgcn  best  1237.712   35.181    20.489   22.88                       Training time    12s/epoch 
		(z-score)                best   1282.160    35.807    20.584   21.86                       Training time    12s/epoch 

Baseline：                                            RMSE        MSE      MAE       MAPE
Transformer   -                                    1791.97    42.332    24.498    31.764                       Training time    2s/epoch

GAT+Transformer (cat x)	          1695.057 41.171   23.683   30.848                      Training time    42s/epoch
![image](https://user-images.githubusercontent.com/49853448/127173903-dc0753ea-d7d9-486a-8658-4fddb4e47032.png)

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
