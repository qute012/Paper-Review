## Transformer Transducer
Transformer transducer 기반 실시간 음성인식기를 만들기위한 공부 내용 정리

#### TRANSFORMER TRANSDUCER: ONE MODEL UNIFYING STREAMING AND NON-STREAMING SPEECH RECOGNITION

기본 구조는 RNN-T와 동일한 구조이다.  라벨 인코더와 오디오 인코더를 joint 해주는 구조인데, joint는 선형 결합하는 구조이다.

![image](https://user-images.githubusercontent.com/33983084/104132938-cdb74400-53c3-11eb-9953-25aa17a5cc9f.png)

Label Encoder는 정확하게 파악하기 힘든데 Y-model에 대한 섹션에서 설명되는 듯 하다. 원 논문을 확인하면 2개의 레이어로 이루어진 라벨 인코더를 사용했다고 하는데, 아마도 LSTM, Linear 이지 싶다. 또한 이전에 논문에서는 LM으로 Transformer를 함께 썼다고 한다.

Fairseq의 논문을 확인해보면 LSTM 2x700과 Transformer를 실험에 사용하였다고 한다. 물론 성능은 Transformer가 압도적이다.

![image](https://user-images.githubusercontent.com/33983084/104133252-b8dbb000-53c5-11eb-868e-879869d9d959.png)



논문에서 설명하는 Transformer 인코더의 구조이다. NLP에서와 마찬가지로 self-attention의 값을 identity와 연결시켜주고 feed forward network가 연결되어 있는 블록을 N번 사용한다.

![image](https://user-images.githubusercontent.com/33983084/104132998-2ab2fa00-53c4-11eb-8154-32acf0c0aa3d.png)

논문에서 **Variable Context Training** 라는 개념이 나온다. 이 역시도 본 논문으로는 이해가 조금 어려운 부분이 있다. 원 논문을 확인하면 attention을 연산하고 나온 결과를 왼쪽과 오른쪽 컨텍스트로 나누어 랜덤하게 마스킹하는데, 이를 랜덤하게 사용하는것 보다 고정적인 값으로 사용할 때 성능이 크게 저하되지 않고 엄청난 속도 개선을 볼 수 있다.

![image](https://user-images.githubusercontent.com/33983084/104135351-1f1aff80-53d3-11eb-96e2-310151ce68be.png)

Full context model(FC)과 Left context model(LC)을 baseline으로 사용하는데, 아마 FC는 패드 토큰 외에 별도의 마스킹을 하지않는 모델이고, LC는 time step마다 1씩 늘려가는 마스킹 기법인 듯 하다. 제안모델인 Y1과 Y2는 오른쪽 컨텍스트를 얼마만큼 사용할지에 따른 벤치마킹인데, Full에 비하면 실시간에서 성능차이가 크게 나지않고, 속도가 32초에서 가장 빠른것은 240ms까지 줄어든다.

![image](https://user-images.githubusercontent.com/33983084/104135525-6655c000-53d4-11eb-9d63-4422ae672452.png)

이미 Transformer의 등장으로 여러 task의 인식률은 굉장히 높다. 꼭 디바이스에서 동작하는 것이 아니여도 모델이 무거워진 만큼 인식속도는 중요하다고 생각이 든다.



# Reference

https://arxiv.org/pdf/1910.12977.pdf

https://arxiv.org/pdf/2002.02562.pdf

https://arxiv.org/pdf/2010.03192.pdf