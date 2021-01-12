## Transformer Implement
Transformer 음성인식 구현을 위한 공부내용 정리

##### 2021-01-13

Transformer를 음성인식으로 학습하기 위한 방법은 크게 두 가지가 있다.

CTC로 학습하거나 Cross-Entropy(Label Smoothing)로 학습, 혹은 두 가지 방법을 섞어서 Joint로 사용하는 방법이 있다.



다음으로 디코딩하여 인식결과를 내기위한 방법도 다르다. 각각 디코딩 방법은 다르다. 마지막에 붙는 Beam Search는 둘 다 동일하게 사용할 수 있다는 점만 같다.  만약 Beam Search의 β를 K라고 가정하면, 먼저 현재 스텝에서 K개의 경로에서 각 노드에서 K개의 후보군을 뽑아낸다. 그러면 총 후보군의 개수는 K*K개가 된다. 그 다음 과정에서 CTC와 CE 디코딩의 차이가 생긴다.

먼저 CE를 사용할 경우에는  K*K개의 후보군에서 hypothesis와 연결될 K개의 노드만 뽑아낸다. CTC와 다르게 중복은 관여하지 않는다.

![image](https://user-images.githubusercontent.com/33983084/104385596-e3209f80-5576-11eb-8670-0d88d217400f.png)

하지만 CTC Prefix Search를 사용하면 K*K 후보군에서 many-to-one map인 B 함수(중복과 Blank 제거)를 통하여서 중복되는 모든 Path를 하나의 노드로 취급하여 hypothesis에 연결된다.

![image](https://user-images.githubusercontent.com/33983084/104385676-01869b00-5577-11eb-9966-a877d23d6323.png)

대조적으로 

# Reference

https://distill.pub/2017/ctc/