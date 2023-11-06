---
title: "Multi Modal Retrieval"
excerpt:

categories: 
    - TOY PROJECT
tags:
    - [Huggingface, NLP, CV, Side Project]

toc: true
toc_sticky: true

date: 2023-10-26
last_modified_at: 2023-11-06
header:
    teaser: "https://github-readme-stats.vercel.app/api/pin/?username=oh-gnues-iohc&repo=multi-modal-retrieval"

    
---


{% linkpreview "https://github.com/oh-gnues-iohc/multi-modal-retrieval" %}


## 프로젝트 개요

CLIP 같은 멀티 모달 도메인의 Retrieval을 직접 구현한 프로젝트

### 개발 이유

CNN이랑 ViT 기반 Image Encoder 성능을 직접 학습해보고 비교 해보고 싶어서 시작

### 개발 목표

- [X] BiEncoder 구현
  - [X] CNN 기반 Image Encoder 구현
  - [ ] ViT 기반 Image Encoder 구현
- [X] 학습

## BiEncoder 구현

Image-Text 유사도 측정은 기존 Text 기반 유사도 측정과 다를게 하나 없음

Image를 CNN이든 ViT던 모델에 태운 뒤, 얻은 정보를 Projection을 통해 Image 임베딩과 크기를 같게 만들어 준 뒤 거리를 구해주면 그게 곧 유사도이니

이번 프로젝트에서는 Text Encoder는 흔히 사용하는 BERT를 사용하고, Image는 ResNet50을 사용하여 Bi-Encoder 구조로 구현할 예정

모델을 만들기 앞서 내가 구현할 모델에 필요한 인자들을 Transformers 라이브러리의 PretrainedConfig를 상속 받아 선언해줘야 함

물론 있는 CLIP Config 가져다 써도 되긴 함

```python
from transformers import ResNetConfig, BertConfig, PretrainedConfig

class ImageTextRetrievalConfig(PretrainedConfig):
    
    model_type = "bert, resnet"
    
    def __init__(
        self, 
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        image_hidden_act="relu",
        downsample_in_first_stage=False,
        out_features=None,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        text_hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        projection_dim=512, 
        logit_scale_init_value=2.6592, 
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        self.text_config = {
            'vocab_size':vocab_size,
            'hidden_size':hidden_size,
            'num_hidden_layers':num_hidden_layers,
            'num_attention_heads':num_attention_heads,
            'intermediate_size':intermediate_size,
            'text_hidden_act':text_hidden_act,
            'hidden_dropout_prob':hidden_dropout_prob,
            'attention_probs_dropout_prob':attention_probs_dropout_prob,
            'max_position_embeddings':max_position_embeddings,
            'type_vocab_size':type_vocab_size,
            'initializer_range':initializer_range,
            'layer_norm_eps':layer_norm_eps,
            'pad_token_id':pad_token_id,
            'position_embedding_type':position_embedding_type,
            'use_cache':use_cache,
            'classifier_dropout':classifier_dropout,
            }
        
        self.image_config = {
            'num_channels':num_channels,
            'embedding_size':embedding_size,
            'hidden_sizes':hidden_sizes,
            'depths':depths,
            'layer_type':layer_type,
            'image_hidden_act':image_hidden_act,
            'downsample_in_first_stage':downsample_in_first_stage,
            'out_features':out_features,
            }

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
```


### Image Encoder

Resnet50을 기준으로 사용할 것이며, Bert와 함께 사용하니 ResnetModel의 last_hidden_state를 projection에 걸어 임베딩을 얻을 생각

```python
image_config = config.image_config
self.image_embed_dim = image_config.hidden_sizes[0]
self.image_encoder = ResNetModel(image_config)
self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
```

### Text Encoder

Text Encoder는 Bert를 사용하고, 마찬가지로 last_hidden_state를 사용하고, projection을 걸 예정

```python
text_config = config.text_config
self.text_embed_dim = text_config.hidden_size
self.text_encoder = BertModel(text_config, add_pooling_layer=False)
self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
```

완성된 모델 코드는 이렇게 나옴

```python
class ImageTextRetrievalPreTrainedModel(PreTrainedModel):
    
    config_class = ImageTextRetrievalConfig
    base_model_prefix = "bert, resnet"
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        factor = self.config.initializer_factor
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)  
            
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
class ImageTextRetrieval(ImageTextRetrievalPreTrainedModel):
    config_class = ImageTextRetrievalConfig
    
    def __init__(self, config: ImageTextRetrievalConfig):
        super().__ini__(config)
        
        text_config = config.text_config
        image_config = config.image_config
        
        self.projection_dim = config.projection_dim
        
        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_sizes[0]
        
        self.text_encoder = BertModel(text_config, add_pooling_layer=False)
        self.image_encoder = ResNetModel(image_config)
        
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        
        self.post_init()
```

학습을 위한 forward 함수는 기존 Bi-Encoder와 마찬가지로 Text(Query), Image(Cadidate) 따로 분리해서 임베딩을 얻은 뒤, loss를 계산해주면 됨

```python
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Tensor = None
    ):
    
    text_embs = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    ).last_hidden_state[:, 0, :]
    
    image_embs = self.image_encoder(
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    ).last_hidden_state[:, 0, :]

    text_embs = self.text_projection(text_embs)
    image_embs = self.image_projection(image_embs)
    
    logit_scale = self.logit_scale.exp()
    logits_per_text = torch.matmul(text_embs, image_embs.t()) * logit_scale
    logits_per_image = logits_per_text.t()
    
    _loss = loss(logits_per_text)
    output = (logits_per_image, logits_per_text, text_embs, image_embs)
    return ((_loss,) + output)
```

## 학습

대조학습(Contrastive learning)을 진행하는 만큼 Batch Size가 성능에 영향을 미치게 됨

근데 또 DPR 논문 보면 무작정 큰건 안좋고, 적당한게 좋다고 하면서 128 썻는데 그 정도 GPU 여유가 없으니 64를 사용

데이터는 Huggingface에서 [diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb)라는 데이터셋을 사용

찾다 보니 실제 사용자가 지정한 프롬프트와 Stable Diffusion에서 생성된 이미지로 구성된 데이터셋이라고 하길래 사용함

컴퓨팅 파워도 있고 용량 문제도 있어서 **2m_random_50k** 사용하기로 결정

```python
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer
import torch
from models.model import ImageTextRetrieval, ImageTextRetrievalConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import TrainingArguments, Trainer, HfArgumentParser
import logging
import os
from dataclasses import dataclass, field
import transformers
from typing import Union


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str=field(
        default="ohgnues/ImageTextRetrieval"
    )
    use_auth_token: str=field(
        default=None, metadata={"help": "비공개 모델 사용에 필요한 인증 토큰"}
    )

@dataclass
class DataArguments:
    path: str=field(
        default="poloclub/diffusiondb", metadata={"help": "데이터셋의 경로 혹은 이름"}
    )
    name: str=field(
        default=None, metadata={"help": "서브셋 이름"}
    )
    cache_dir: str=field(
        default=None, metadata={"help": "캐시 파일 저장 위치"}
    )
    train_split: str = field(
        default="train", metadata={"help": "학습 데이터 이름"}
    )
    eval_split: str = field(
        default=None, metadata={"help": "평가 데이터 이름"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "데이터 셔플 여부"}
    )
    text_column_name: str = field(
        default="prompt", metadata={"help": "Text 데이터 Column 이름"}
    )
    image_column_name: str = field(
        default="image", metadata={"help": "Image 데이터 Column 이름"}
    )
    max_length: int = field(
        default=512, metadata={"help": "최대 토큰 길이"}
    )


@dataclass
class TrainArguments(TrainingArguments):
    output_dir: str = "runs/"
    do_train: bool = True
    do_eval: bool = False
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 5.0
    learning_rate: float = 5e-5
    save_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = 'epoch'
    



if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    model = ImageTextRetrieval.from_pretrained(**vars(model_args))
    tokenizer = AutoTokenizer.from_pretrained(**vars(model_args))
    processor = AutoImageProcessor.from_pretrained(**vars(model_args))

    if os.path.isdir(data_args.path):
        dataset = load_from_disk(data_args.path)
    else:
        dataset = load_dataset(data_args.path, data_args.name, cache_dir=data_args.cache_dir)

    if data_args.shuffle:
        dataset = dataset.shuffle()


    def example_function(examples):

        tokenized_text = tokenizer(
            examples[data_args.text_column_name],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_length,
            return_tensors="pt"
        )

        processed_image = processor(examples[data_args.image_column_name], return_tensors="pt")

        tokenized_text.update(processed_image)

        return tokenized_text

    dataset = dataset.map(example_function, batched=True, batch_size=10000, remove_columns=dataset[data_args.train_split].column_names)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split] if data_args.eval_split else None,
    )

    trainer.train()

```

이렇게 해서 학습은 총 10 에폭으로 마무리 되었음

```bash
{'train_runtime': 45238.1827, 'train_samples_per_second': 11.053, 'train_steps_per_second': 0.173, 'train_loss': 9.051637022208679, 'epoch': 10.0}
```

실 사용을 위해선 forward가 아닌 함수를 따로 구현 해야함

retireval의 목적 특히나 Bi-Encoder 구조의 목적은 수 많은 데이터를 미리 임베딩 해 Document pool을 구축하는데에 있음

```python
    def encode(self, model_name: Literal["text", "image"],
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Tensor = None
            ):
        
        if model_name == "text":
            self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ).last_hidden_state[:, 0, :]
        
        elif model_name == "image":
            self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            ).pooler_output[:, :, 0, 0]
```

이제 이 함수를 통해 배치 단위의 데이터 혹은 단일 데이터의 임베딩을 얻을 수 있음

이렇게 얻은 임베딩들 사이의 거리를 `torch.matmul`을 이용해 구하면 해당 모델을 완벽하게 사용할 수 있음