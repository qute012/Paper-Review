## Transformer Implement
Transformer 음성인식 구현을 위한 공부내용 정리

##### 2021-01-13

Transformer를 음성인식으로 학습하기 위한 방법은 크게 두 가지가 있다.

CTC로 학습하거나 Cross-Entropy(Label Smoothing)로 학습, 혹은 두 가지 방법을 섞어서 Joint로 사용하는 방법이 있다. 마지막에 FC layer를 사용하여 각 캐릭터의 probabilities를 계산하는데, 이때 CE는 End-to-End가 아닌 기존 학습 방법대로 사용하면되는데, 이때 예측은 loss를 계산하기 위해 target과 똑같은 길이의 time step으로 padding된 채로 나오는 것으로 예상된다(왜냐하면 nn.CTCLoss 처럼 시퀀스의 원래 사이즈를 안 받기 때문). 아직은 코드 리뷰 중이여서 아마 직접 구현을 해보면서 알게될 것 같다. CTC loss는 이론은 공부를 하였는데 코드와 함께 리뷰는 하지않아서 일단은 생략했다(사실 function으로 넣으면 척 나오기 때문이 큼).

다음으로 디코딩하여 인식결과를 내기위한 방법도 다르다. 각각 디코딩 방법은 다르다. 마지막에 붙는 Beam Search는 둘 다 동일하게 사용할 수 있다는 점만 같다.  만약 Beam Search의 β를 K라고 가정하면, 먼저 현재 스텝에서 K개의 경로에서 각 노드에서 K개의 후보군을 뽑아낸다. 그러면 총 후보군의 개수는 K*K개가 된다. 그 다음 과정에서 CTC와 CE 디코딩의 차이가 생긴다.

먼저 CE를 사용할 경우에는  K*K개의 후보군에서 hypothesis와 연결될 K개의 노드만 뽑아낸다. CTC와 다르게 중복은 관여하지 않는다.

![image](https://user-images.githubusercontent.com/33983084/104385596-e3209f80-5576-11eb-8670-0d88d217400f.png)

하지만 CTC Prefix Search를 사용하면 K*K 후보군에서 many-to-one map인 B 함수(중복과 Blank 제거)를 통하여서 중복되는 모든 Path를 하나의 노드로 취급하여 hypothesis에 연결된다. 이론적으로는 B 함수인데 코드상으로는 prev_token과 중복을 보는 듯 하다.

![image](https://user-images.githubusercontent.com/33983084/104385676-01869b00-5577-11eb-9966-a877d23d6323.png)

현재 서브 레이어 구현은 완료하였으나, Transformer의 학습과 디코딩 구현을 위해 좀 더 공부가 필요할 것 같다.



# Reference

https://distill.pub/2017/ctc/