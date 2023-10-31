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
last_modified_at: 2023-10-31
header:
    teaser: "https://github-readme-stats.vercel.app/api/pin/?username=oh-gnues-iohc&repo=multi-modal-retrieval"

    
---


{% linkpreview "https://github.com/oh-gnues-iohc/multi-modal-retrieval" %}


## 프로젝트 개요

CLIP 같은 멀티 모달 도메인의 Retrieval을 직접 구현한 프로젝트

### 개발 이유

해보고싶어서, CNN이랑 ViT 기반 Image Encoder 성능을 직접 학습해보고 비교 해보고 싶어서

### 개발 목표

- [X] BiEncoder 구현
  - [X] CNN 기반 Image Encoder 구현
  - [ ] ViT 기반 Image Encoder 구현
- [ ] 학습

## BiEncoder 구현

Image-Text 유사도 측정은 기존 Text 기반 유사도 측정과 다를게 하나 없음

Image를 CNN이든 ViT던 태운 뒤, 얻은 정보를 Projection을 통해 Image 임베딩과 크기를 같게 만들어 준 뒤 거리를 구해주면 그게 곧 유사도이니

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