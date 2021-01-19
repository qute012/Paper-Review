## 0. Few-shot Learning studying
#### Opinion

카카오 브레인 AutoLearn팀에서 진행한 연구이다. 먼저 퓨샷 러닝의 개념을 짚고 넘어가면, N-way K-shot 문제이다. 예전에 음성 명령어 인식 용 PrototypeNet을 학습해 본 경험이 있는데, 이 때는 문제를 조금 잘 못 이해하고 사용하였다. N은 범주의 수, K는 범주별 서포트 데이터의 수인데, 일반적으로 퓨샷 러닝에서 이러한 평가 기법을 학습 데이터에 등장하지 않은 클래스를 분류할 때 사용하는데, 나는 그냥 적은 양의 학습 데이터에서 성능을 높이기 위해 사용하였었다. 물론 문제 해결에도 나름 도움이 되었다. ResNet보다 더 가벼운 모델 구조를 사용하여도 성능이 월등히 높았다.



#### Trend (refered kaKaobrain)

퓨샷 러닝의 트렌드는 2가지로 나뉨.

1. 거리 학습 기반 방식
   서포트 데이터와 쿼리 데이터 간의 거리를 측정하는 방식을 활용
2. 그래프 신경망 방식
   일반적인 DNN의 입력은 벡터나 행렬 형태를 활용한다면, GNN은 밀집 그래프 구조를 활용



본 리뷰에서는 GNN의 구조에 대한 해석과 Kakao brain에서 제안된 기법이 뭔지 살펴보려 한다.

#### GNN Preview (refered kakaobrain)

그래프 기반의 퓨샷 러닝은 두 가지 유형의 동향을 보임. 여기서는 GNN만 공부할 예정.

1. **[Few-shot learning with graph neural networks](https://arxiv.org/abs/1711.04043)**

   각 노드는 해당하는 데이터의 특징 벡터로 초기화 된다. 이 부분이 무슨 뜻인지 이해가 되지 않아서 코드를 함께 보았다. 전체 GNN 구조는 아래와 같이 되어 있음. 

   ![image](https://user-images.githubusercontent.com/33983084/104985015-55035800-5a53-11eb-9ff2-3f747c715977.png)

   특정 노드 V의 이웃 노드에 노드별 유사도를 곱한 값들의 합을 구하여 이를 V와 합쳐 새로운 벡터 V'를 얻는다.

   ```
   def forward(self, x):
           X_i = x.unsqueeze(2) # (b, N , 1, input_dim)
           X_j = torch.transpose(X_i, 1, 2) # (b, 1, N, input_dim)
   
           phi = torch.abs(X_i - X_j) # (b, N, N, input_dim)
   
           phi = torch.transpose(phi, 1, 3) # (b, input_dim, N, N)
   
           A = phi
   
           for l in self.module_list:
               A = l(A)
           # (b, 1, N, N)
   
           A = torch.transpose(A, 1, 3) # (b, N, N, 1)
   
           A = F.softmax(A, 2) # normalize
   
           return A.squeeze(3) # (b, N, N)
   ```

   아직 코드 전체를 해석한 것은 아닌데, 특징 벡터란 CNN Extractor에서 추출된 벡터를 뜻하는 듯 하고, 이러한 특징 벡터와 원래 입력인 x와 곱하여 선형 분류하는 것이 Graph Convolution 인듯함.

   ```
   def gmul(input):
       W, x = input
       # x is a tensor of size (bs, N, num_features)
       # W is a tensor of size (bs, N, N, J)
       x_size = x.size()
       W_size = W.size()
       N = W_size[-2]
       W = W.split(1, 3)
       W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
       output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
       output = output.split(N, 1)
       output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
       return output
   ```

   위의 두 가지 연산을 순차적으로 수행하여 가장 마지막에 쿼리 노드 벡터값의 업데이트도 완료한다. 모델은 N개의 범주와 완전히 연결된 FC 층을 통해 쿼리 데이터의 범주를 예측함

   여기서 거리 학습 방식과의 차이점이 보인다. 최종 분류 과정에서 유사도를 사용하는 것이 아니라, 레이어의 학습에 유사도를 사용하여, 각 레이어가 노드 간의 상관 관계를 학습하는 걸로 이해됨.
   
   다음으로는 Edge-Labeling Graph Neural Network for Few-shot Learning를 공부할 예정이다.

# Reference

https://www.kakaobrain.com/blog/112