## Abstract
ContextNet은 librispeech에서 현재 SOTA 모델이다. 성능은 wer clean에서 1.9%, other에서 4.1%이다.
본 모델을 사용해서 Noisy Pretrained Model을 사용한 슈도라벨링을 사용하여 wer을 1.7%까지 올린 연구도 있다.
모델을 간단하게 설명하면 CNN Extractor를 사용하여 음성의 Representation을 뽑아낸 뒤, RNN으로 Transduce하여 각 음소를 인식하는 구조이다.

## Model Architecture
모델은 Encoder와 Decoder로 이루어져있다. Encoder는 여러층의 CNN으로 이루어져있고, Decoder는 LSTM으로 이루어져있다.
인코딩 된 피쳐를 디코더의 LSTM이 매 타입스텝별로 캐릭터의 represent하게되고 최종적으로 선형 분류를 거쳐 시퀀스를
생성한다.

#### Encoder
일반적으로 음성인식 모델은 입력으로 음성의 signal을 spectrogram으로 변환하여 학습하게 되는데,
ContextNet은 siganl을 입력으로 받게된다. 그리고 겹겹의 cnn을 거쳐 high level representation으로 변환된다.

#### -Squeeze-and-excitation(SE)
벡터를 각 채널별로 합을 구한 값에 각 채널이 가진 시간적 길이를 나누어 커널을 global하게 사용하여,
average pooling과 같은 효과를 적용한 후, 여기에 sigmoid를 적용하여 원래 identity인 x와 곱하는
레이어이다. 예를 들면, mini-batch가 16, 채널이 80 그리고 타입스텝이 400인 벡터가 있을 때, x는 (16,400,80)
shape이다. 그러면 80의 각 채널을 먼저 더하게 되면 x'는 (16,80)의 구조가 되고, x'에 x의 배치마다의 길이를 나누어
 평균을 구한후 가중치 W를 곱한 후 확장하여 (16,1,80)의 크기가 되고 거기에 sigmoid를 적용한 후 x = x * x' 

#### -Swish activation function
기존의 ReLU는 0이하는 0, 0이상은 identity로 연산하는 방법인데, 
![image](https://user-images.githubusercontent.com/33983084/94438247-d3549980-01d9-11eb-96f8-d761632c838e.png)
x를 1+x^2으로 나누어주는 방법인데, 음수일 경우에도 매우 작은 음수 값이 나온다.

#### -Depthwise seprable convolution(DS Conv)
이 방법은 일반적인 Convolution Layer의 파라미터를 줄이는 방법인데, 채널을 먼저 kx1 kernel로 합성곱을 하고
시간영역을 후에 1 kernel을 사용하여 합성곱을 하는 방법이다. 예를 들어 Conv2d(3, 80, (3,3))을
Conv2d(3, 80, (3,1))과 Conv2d(80, 80 (1,1))로 사용한다.

#### -Convolution block
x를 겹겹히 쌓은 Conv-BN로 연산하고 위의 SE 레이어를 사용한 후 skip connection을 사용하여 pointwise conv1d를
사용

#### -Progrssive downsampling
stride를 2로사용하여 sample의 개수를 줄인다.

## Performance
![image](https://user-images.githubusercontent.com/33983084/94435342-fc732b00-01d5-11eb-8364-f2e3a24daeb2.png)


## Reference
ContextNet: https://arxiv.org/pdf/2005.03191.pdf
Improved Noisy Student Training for Automatic Speech Recognition: https://arxiv.org/pdf/2005.09629v1.pdf
