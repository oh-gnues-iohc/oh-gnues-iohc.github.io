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
last_modified_at: 2023-11-06
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

Generator의 경우 아래 코드처럼 구현했음

```python
class GeneratorPreTrainedModel(PreTrainedModel):
    
    config_class = GeneratorConfig
    base_model_prefix = "gan"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
class Generator(GeneratorPreTrainedModel):
    
    def __init__(self, config:GeneratorConfig):
        super().__init__(config)
        img_shape = (config.img_size ** 2) * config.img_channels
        
        self.config = config
        self.encoder = GanEncoder(config)
        self.projection = nn.Linear(128 * 2 ** config.num_layer, img_shape)
        self.act = nn.Tanh()
        
        self.post_init()
        
    def forward(self, input: Optional[torch.Tensor]):
        embeddings = self.encoder(input)
        image = self.projection(embeddings)
        image = self.act(image)
        
        return image.view(input.size(0), self.config.img_channels, self.config.img_size, self.config.img_size)

        
class GanEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = GanLayer(config.latent_dim, 128, config.activation, normalize=False)
        self.layer = nn.ModuleList([GanLayer(128 * 2 ** i, 128 * 2 ** (i+1), config.activation) for i in range(config.num_layer)])
        
    def forward(self, input: Optional[torch.Tensor]):
        output = self.head(input)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
        
class GanLayer(nn.Module):
    
    def __init__(self, in_feat: int, out_feat: int, activation: str, normalize: bool=True):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = nn.functional.gelu if activation == "gelu" else (nn.ReLU if activation == "relu" else nn.LeakyReLU)
        self.layer = nn.Linear(in_feat, out_feat)
        self.norm = None
        if normalize:
            self.norm = nn.LayerNorm(out_feat, eps=1e-12)
            
    def forward(self, input: Optional[torch.Tensor]):
        output = self.layer(input)
        if self.norm:
            output = self.norm(output)
        return self.activation(output)
```

하나하나 설명해보자면 우선 GanEncoder

```python
class GanEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = GanLayer(config.latent_dim, 128, config.activation, normalize=False)
        self.layer = nn.ModuleList([GanLayer(128 * 2 ** i, 128 * 2 ** (i+1), config.activation) for i in range(config.num_layer)])
        
    def forward(self, input: Optional[torch.Tensor]):
        output = self.head(input)
        for i, layer_module in enumerate(self.layer):
            output = layer_module(output)
        return output
```

이 모듈의 역할은 Retrieval에서 얻어온 Text embedding을 입력 받고, 해당 벡터에서 feature를 뽑는 역할



BN vs LN