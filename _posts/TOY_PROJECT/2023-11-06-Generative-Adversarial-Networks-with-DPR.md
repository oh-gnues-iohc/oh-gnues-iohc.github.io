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

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/26499fba-a148-490b-a9f9-f82e81a45dfd)

GAN도 원리는 간단함 Generator와 Discriminator로 이루어지는데, Generator는 노이즈에서 이미지를 생성하고, Discriminator는 원본 이미지와 Generator가 생성한 이미지를 분류하는 방식 즉 적대적으로 학습되는 아이디어임

GAN은 학습 방식 상 이미지를 생성하는 모델과 이미지를 판별하는 모델 모두 학습이 되기 때문에 Generator 뿐만 아니라 적은 데이터로도 좋은 성능의 Classifier를 얻고 싶을 때도 사용함

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/3e6e73e6-dbca-42fc-91b8-8e3e7d836a08)


그럼 이렇게 좋은 GAN을 왜 요즘엔 잘 안쓰고 Diffusion 모델이 뜰까라고 묻는다면 답은 간단함

GAN은 Diffusion 처럼 원하는 이미지를 생성할 수 없음

좀 더 정확히 말하면 원하는 이미지를 **상세히** 생성할 수 없음

GAN은 Diffusion 모델과 다르게 설정 되어있는 클래스에 맞춰 대략적인 이미지밖에 생성할 수 없음

코끼리를 생성하는건 가능하지만 물구나무선 코끼리를 원트에 딱 찝어서 생성하는건 불가능

Generator를 학습할 때 Loss가 되는 Discriminator가 말했듯 Classifier이기 때문에 일어나는 현상임 물구나무선 코끼리를 GAN으로 생성하기 위해선 `물구나무선 코끼리`라는 Label을 지정해야함

그럼 여기서 GAN이 Diffusion처럼 유연하게, 상세하게 이미지를 생성하지 못하는 이유가 Discriminator를, Classifier를 Loss로 사용하기 때문이라면 Loss를 DPR로 지정하여 적대적 학습을 진행한다면?

GAN이 생성한 이미지를 Hard Negative로 지정하여 DPR을 학습한다면??

Retrieval 또한 더 좋은 성능으로 학습을 할 수 있는게 아닐까? 아니더라도 GAN으로 Diffusion 처럼 원하는 이미지를 생성할 수 있는게 아닐까???

### 개발 목표