---
title: "Generative Adversarial Networks with DPR"
excerpt:

categories: 
    - TOY PROJECT
tags:
    - [Huggingface, NLP, CV, Side Project]

toc: true
toc_sticky: true

date: 2023-11-06
last_modified_at: 2023-11-14
header:
    teaser: "https://github-readme-stats.vercel.app/api/pin/?username=oh-gnues-iohc&repo=Generative-Adversarial-Networks-with-DPR"

    
---


{% linkpreview "https://github.com/oh-gnues-iohc/Generative-Adversarial-Networks-with-DPR" %}


## 프로젝트 개요

DPR(Dense Passage Retrieve)을 이용한 생성 제어가 가능한 GAN(Generative Adversarial Network) 모델

### 개발 이유

요즘 이미지를 생성하는 모델은 Diffusion 모델을 많이 사용한다. 근데 Diffusion 모델 말고 GAN 이라는 녀석도 데이터를 생성할 수 있고, 실제로도 많이 쓰였음

GAN은 Diffusion이랑 뭐가 다르냐 우선 Diffusion 부터 알아보자

Diffusion의 원리는 간단함 원본 이미지에 노이즈를 조금씩 덮은 뒤, 디노이징하는 모델을 학습하면 완전한 노이즈에서 원하는 이미지를 생성(디노이징의 반복)할 수 있는 모델이 된다는 아이디어

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/b50dfb63-8994-4512-8c14-342e1b8d346a)

즉 이렇게 학습을 한 뒤, 사용할 때는 랜덤 노이즈와 함께 "이 노이즈는 사실 어떤 외눈박이의 사진인데, 노이즈가 너무 많이 끼었어" 라고 입력하면, 모델이 알아서 디노이징을 해주는 것

<p align="center"><img src="https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/26499fba-a148-490b-a9f9-f82e81a45dfd" width="50%" height="50%"/></p>

GAN도 원리는 간단함 Generator와 Discriminator로 이루어지는데, Generator는 노이즈에서 이미지를 생성하고, Discriminator는 원본 이미지와 Generator가 생성한 이미지를 분류하는 방식 즉 적대적으로 학습되는 아이디어임

GAN은 학습 방식 상 이미지를 생성하는 모델과 이미지를 판별하는 모델 모두 학습이 되기 때문에 Generator 뿐만 아니라 적은 데이터로도 좋은 성능의 Classifier를 얻고 싶을 때도 사용함

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/3e6e73e6-dbca-42fc-91b8-8e3e7d836a08)


그럼 이렇게 좋은 GAN을 왜 요즘엔 잘 안쓰고 Diffusion 모델이 뜰까라고 묻는다면 답은 간단함

GAN은 Diffusion 처럼 원하는 이미지를 생성할 수 없음 좀 더 정확히 말하면 원하는 이미지를 **상세히** 생성할 수 없음

GAN은 Diffusion 모델과 다르게 설정 되어있는 클래스에 맞춰 대략적인 이미지밖에 생성할 수 없음 즉, 코끼리를 생성하는건 가능하지만 물구나무선 코끼리를 원트에 딱 찝어서 생성하는건 불가능

Generator를 학습할 때 Loss가 되는 Discriminator가 말했듯 Classifier이기 때문에 일어나는 현상임 물구나무선 코끼리를 GAN으로 생성하기 위해선 `물구나무선 코끼리`라는 Label을 지정해야함

그럼 여기서 GAN이 Diffusion처럼 유연하게, 상세하게 이미지를 생성하지 못하는 이유가 Discriminator를, Classifier를 Loss로 사용하기 때문이라면 Loss를 DPR로 지정하여 적대적 학습을 진행한다면?

GAN이 생성한 이미지를 Hard Negative로 지정하여 DPR을 학습한다면??

Retrieval 또한 더 좋은 성능으로 학습을 할 수 있는게 아닐까? 아니더라도 GAN으로 Diffusion 처럼 원하는 이미지를 생성할 수 있는게 아닐까???

<p align="center"><img src="https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/f05bbb1b-b03c-4e7c-b3d7-48c210678709" width="50%" height="50%"/></p>

### 개발 목표

### 구조

구상한 구조는 아래 그림과 같음

1. 프롬프트를 Text Encoder에 태워 얻은 Text Embedding을 Generator의 입력으로 넣어 이미지를 생성

<p align="center"><img src="https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/ce9eb5d6-07b0-4afc-b53b-db521a2c17de" width="50%" height="50%"/></p>

2. 생성한 이미지와 원본 이미지를 Image Encoder에 태워 Image Embedding을 얻은 뒤, 두 벡터 사이의 거리를 Generator의 Loss로 설정하여 학습

<p align="center"><img src="https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/c1cc0212-838a-4625-b6cb-d7fe8682c0f2" width="50%" height="50%"/></p>

3. 생성한 이미지를 Hard Negative로 설정하여 DPR 학습

<p align="center"><img src="https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/fc6535f1-da8c-4406-866d-71c19d9bf326" width="50%" height="50%"/></p>


물론 이 구조를 사용할 경우 프롬프트 하나당 한개의 이미지만 생성 가능하지, 다양성이 존재하는 이미지를 생성할 수는 없음

이 문제는 나중에 따로 Conditional GAN에서 사용된 conditional을 적용하여 해결할 예정

### 구현

구조는 이렇게 정했으니 이제 구현을 할 차례 Generator와 Discriminator를 구현해야 하는데, 후자는 이미 구현한 [Multi Modal Retrieval](https://oh-gnues-iohc.github.io/toy%20project/multi-modal-retrieval/)을 사용할 것

#### Generator

Generator의 경우 아래 코드처럼 DCGAN 구조로 구현했음

```python
import transformers
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from models.config import GeneratorConfig
from typing import Optional, Literal
from utils.train_utils import kaiming_normal_
class GeneratorPreTrainedModel(PreTrainedModel):
    
    config_class = GeneratorConfig
    base_model_prefix = "gan"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="gelu")
            
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)  
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
class Generator(GeneratorPreTrainedModel):
    
    def __init__(self, config:GeneratorConfig):
        super().__init__(config)
        self.config = config
        self.encoder = GanEncoder(config)
        self.projection = nn.Conv2d(64, config.img_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.Tanh()
        
        self.post_init()
        
    def forward(self, input: Optional[torch.Tensor]):
        embeddings = self.encoder(input)
        image = self.projection(embeddings)
        image = self.act(image)
        
        return image


class GanEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_size = config.img_size // (2 ** config.num_layer)
        self.init_dim = (64 * (2 ** config.num_layer))
        self.head = nn.Linear(config.latent_dim, self.init_dim * self.init_size ** 2)
        self.layer = nn.ModuleList([GanLayer(in_channels=64 * (2 ** (config.num_layer - i)), 
                                             out_channels=64 * (2 ** (config.num_layer - i - 1)), 
                                             activation = config.activation) for i in range(config.num_layer)])
        
    def forward(self, input: Optional[torch.Tensor]):
        output = self.head(input)
        output = output.view(output.shape[0], self.init_dim, self.init_size, self.init_size)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
        
class GanLayer(nn.Module):
    
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.functional.gelu if activation == "gelu" else (nn.functional.relu if activation == "relu" else nn.functional.leaky_relu)
        self.upsample = nn.Upsample(scale_factor=2)
        self.block = self.conv_block(in_channels, out_channels, kernel_size, stride)
        self.block2 = self.conv_block(out_channels, out_channels, kernel_size, stride)
        
    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
            
    def forward(self, input: Optional[torch.Tensor]):
        _x = self.upsample(input)
        _x = self.block(_x)
        _x = self.block2(_x)
        return self.activation(_x)
```

하나하나 설명해보자면 우선 _init_weights 부분을 보면

```python
def _init_weights(self, module):
    if isinstance(module, nn.Conv2d):
        kaiming_normal_(module.weight, mode="fan_out", nonlinearity="gelu")
```

kaiming_normal_을 통해 Conv 레이어를 초기화 하는데, DCGAN 논문에는 평균 0, 표준편차 0.02로 초기화를 해주라고 나와있음

하지만 옛날 논문이기도 해서 둘 다 실험해볼 생각

다음으로 GanEncoder

```python
class GanEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_size = config.img_size // (2 ** config.num_layer)
        self.init_dim = (64 * (2 ** config.num_layer))
        self.head = nn.Linear(config.latent_dim, self.init_dim * self.init_size ** 2)
        self.layer = nn.ModuleList([GanLayer(in_channels=64 * (2 ** (config.num_layer - i)), 
                                             out_channels=64 * (2 ** (config.num_layer - i - 1)), 
                                             activation = config.activation) for i in range(config.num_layer)])
        
    def forward(self, input: Optional[torch.Tensor]):
        output = self.head(input)
        output = output.view(output.shape[0], self.init_dim, self.init_size, self.init_size)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
```

이 모듈은 Retrieval에서 얻어온 Text embedding을 입력 받고, 해당 벡터에서 feature를 뽑는 역할을 함

head는 num_layer에 비래하여 init_size를 지정해주고, 그 수만큼 Linear layer를 통해 확장해주는 역할

그 뒤 CNN을 통해서 Upsample, Conv를 적용하는게 layer인데, 이 모듈은 모두 `GanLayer`로 이루어져 있음

```python
class GanLayer(nn.Module):
    
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.functional.gelu if activation == "gelu" else (nn.functional.relu if activation == "relu" else nn.functional.leaky_relu)
        self.upsample = nn.Upsample(scale_factor=2)
        self.block = self.conv_block(in_channels, out_channels, kernel_size, stride)
        self.block2 = self.conv_block(out_channels, out_channels, kernel_size, stride)
        
    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )
            
    def forward(self, input: Optional[torch.Tensor]):
        _x = self.upsample(input)
        _x = self.block(_x)
        _x = self.block2(_x)
        return self.activation(_x)
```

GanLayer는 DCGAN 구조를 참고하여 만들었음

Upsample, Conv, Norm, Act 순서로 피드포워딩 하는 모듈

여기서 GanLayer 하나당 Upsample 하나이기 때문에 `self.init_size = config.img_size // (2 ** config.num_layer)`가 되는것

##### BN vs LN vs GN

Generator 모델 구조를 만드는 도중 Batch Norm과 Layer Norm 중 무엇을 써야할지 고민이 됨

###### Batch Normalization

- 장점
  - Overfitting 감소
  - 안정적인 학습, 빠른 수렴
  - 피드포워드 네트워크 모델(한방향으로 흐르는 인공신경망 모델)에 적합
  
- 단점
  - 미니배치 크기에 의존
  - 시계열 모델에 적용 어려움

###### Layer Normalization

- 장점
  - 작은 배치에도 적용 가능
  - 시계열 모델에 적용 가능
  - 일반화 성능 향상 가능
  
- 단점
  - 추가 계산, 오버헤드 발생
  - 피드포워드 모델에 적합하지 않음

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/669145f4-89f6-4f2c-8c07-4a8d091f8f95)

실제로 RNN 모델에서는 BN 과 LN의 차이가 상당히 남

이런 이유로 보통 CV에서는 BN을 사용하고, NLP에서는 LN을 사용하는게 약간 국룰 느낌으로 굳어졌는데, Convnet 2020s 논문을 보면 CV에서도 LN이 미약하게나마 성능이 좋다고는 하는데 잘 모르겠음

그런데, 학습 데이터 총 50만 건중 배치의 크기가 너무 작아 배치의 평균과 분산이 데이터셋 전체를 대표한다는 가정을 만족시키기 어려움

때문에 BN과 GN을 비교해보기로 함

### 학습

학습에 있어 문제가 있음

GAN의 고질적인 문제, 학습이 어려운 여러 이유중 하나인 힘의 균형이 있음

당연하게도 분류기 보다 생성기를 학습시키는 것이 일반적으로 어려움 일반적으로 판별자가 생성기보다 먼저, 잘 학습이 되는데 이때 둘 사이의 힘의 균형이 깨지는 경우 GAN 학습이 더이상 진전될 수 없음

즉 티키타카가 잘 되어야 애프터, 삼프터가 되는데 판별자가 첫만남에 급발진으로 고백을 박아버리는 문제가 생김

흔한 GAN 문제이지만, 해당 프로젝트에서는 아주 아주 큰 문제로 발생하게 됨

그 이유는 문장을 임베딩하는 본 프로젝트의 판별자 즉 DPR(Retrieval)의 경우 사전학습된 모델을 사용하지 않으면 아주 많은 데이터셋과, 아주 많은 컴퓨팅 파워로 오랜 기간을 들여 학습을 시켜야 함

그렇지 않고 사전학습된 모델을 사용할 경우 생성자와 판별자가 Init Weight에서 시작하는 것보다 판별자가 더 빨리 수렴해버림 사전 학습된 판별자에세 초기화된 생성자가 생성하는 노이즈 가득한 이미지는 누워서 떡먹기임

사전학습할 자원도, 여유도 없으니 Text Encoder 즉 BERT만 사전학습된 모델을 사용하고 Image Encoder ResNet은 Init Weight로 생성자와 함께 학습 시키기로 결정


11.14 - 학습이 너무 안됨 이것저것 바꿔봐도 답이 안보임