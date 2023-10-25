---
title: "Huggigface Trainer Arguments Viewer"
excerpt:

categories: 
    - TOY PROJECT
tags:
    - [Huggingface, Python, Side Project]

toc: true
toc_sticky: true

date: 2023-10-23
last_modified_at: 2023-10-23
header:
    teaser: "https://github-readme-stats.vercel.app/api/pin/?username=oh-gnues-iohc&repo=Huggingface-Trainer-Args-Viewer"

    
---


{% linkpreview "https://github.com/oh-gnues-iohc/Huggingface-Trainer-Args-Viewer" %}


## 프로젝트 개요

HuggingFace의 Trainer의 인자(Arguments) 관리를 쉽게 하기 위한 프로젝트

### 개발 이유

GPT, BERT, T5, ... 여러 모델과 모델의 목적에 따라 사용되는 인자가 너무 많아 관리가 어려움

인자가 많아짐에 따라 실행 명령어를 관리하는데에 어려움이 있음

인자를 한눈에 보기 어려움

### 개발 목표

- [X] [Dataclass 추출](#dataclass-추출)
  - [X] [Argument 추출](#argument-추출)
- [ ] [Streamlit 구축](#streamlit-구축)
  - [ ] 옵션 설정 기능 추가
- [ ] 설정한 옵션들로 Python 실행 명령어 출력



## Dataclass 추출

AST: Python Abstract Syntax Tree

Python에서 기본 제공하는 package

이름에서 알 수 있듯 Python 코드를 넣으면, Syntax를 쉽게 분석 할 수 있는 Tree가 나오게 됨

Dataclass를 추출하기 위해선 `@dataclass` 데코레이터를 찾으면 쉬움

```python
@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str=field(
        default=""
    )
    use_auth_token: str=field(
        default="", metadata={"help": "비공개 모델 사용에 필요한 인증 토큰"}
    )
```

`ast.parse` 함수를 이용해 코드에 대한 tree를 얻을 수 있음

이 tree를 이용해 각 node에 접근이 가능

```python
with open(file_path, 'r', encoding='utf-8') as file:
    tree = ast.parse(file.read())

for node in ast.walk(tree):
    ...
```

각 노드들은 `ast.Name`, `ast.ClassDef`, `ast.Call` 등 여러 instance로 존재함

자세한 내용은 [공식 문서](https://docs.python.org/3/library/ast.html)를 참고하고, 여기선 `ast.ClassDef` 만 사용하면 됨

`isinstance(node, ast.ClassDef)` 에 걸리는 node들은 모두 Class이니, 여기서 `@dataclass`라는 데코레이터를 가지고 있는 node만 다시 걸러줘야함

`node.decorator_list`를 사용하면 해당 노드가 가지고 있는 모든 데코레이터를 얻을 수 있음

```python
with open(file_path, 'r', encoding='utf-8') as file:
    tree = ast.parse(file.read())

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        decorators = {d.id for d in node.decorator_list}
        if "dataclass" in decorators:
            print(node.name)
```

```
ModelArguments
DataArguments
TrainArguments
```

### Argument 추출

각 Dataclass들을 추출했으면 다음은 Argument를 추출할 차례

`ast.ClassDef` 문서를 보면 아래와 같음

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/f56f2424-06a5-43d4-b00b-ab0c83dd9735)

`body`는 클래스 내부에 정의된 코드의 노드임 즉 이걸 통해서 Argument를 가져올 수 있음

```python
class DataclassFinder(list):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
        tree = self._parse_file()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and self._has_dataclass_decorator(node):
                dataclass = self._parse_dataclass(node)
                self.append(dataclass)

    def _parse_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return ast.parse(file.read())

    def _has_dataclass_decorator(self, class_node):
        decorators = {d.id for d in class_node.decorator_list}
        return "dataclass" in decorators

    def _parse_dataclass(self, class_node):
        dataclass = {"name": class_node.name, "elements": []}
        for class_element in class_node.body:
            if isinstance(class_element, ast.AnnAssign):
                element = self._parse_class_element(class_element)
                dataclass["elements"].append(element)
        return dataclass

    def _parse_class_element(self, class_element):
        element = {"name": class_element.target.id, "type": None, "default": None, "help": None}
        if isinstance(class_element.annotation, ast.Name):
            element["type"] = class_element.annotation.id
        if class_element.value and isinstance(class_element.value, ast.Call):
            for keyword in class_element.value.keywords:
                if keyword.arg == 'default':
                    element["default"] = ast.literal_eval(keyword.value)
                elif keyword.arg == "metadata":
                    element["help"] = ast.literal_eval(keyword.value)["help"]
        return element

```

## Streamlit 구축
