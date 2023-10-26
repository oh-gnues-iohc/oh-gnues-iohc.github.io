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
last_modified_at: 2023-10-26
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

- [X] Dataclass 추출
  - [X] Argument 추출
- [X] Streamlit 구축
  - [X] 옵션 설정 기능 추가
- [X] 설정한 옵션들로 Python 실행 명령어 출력



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

Streamlit은 Python을 이용해서 빠르게 어플리케이션을 만들 수 있는 라이브러리

### 파일 읽기

`st.file_uploader`를 이용하여 쉽게 파일을 읽을 수 있음

```python
uploaded_file = st.file_uploader("Choose a Python file", accept_multiple_files=False)
if uploaded_file:
    st.write(DataclassFinder(uploaded_file.read()))
```

![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/c85963fb-dbba-4a0f-a2a7-7f82913a74f0)

이렇게 streamlit을 통해 얻은 파일의 Dataclass를 추출하였으니, 이걸 편하게 Display하면 됨

전체적인 Streamlit 코드는 아래와 같음

```python
from srcs.finder import DataclassFinder
import streamlit as st

command = {}
setter = {"str": str, "int": int, "float": float, "bool": bool}


def main():
    uploaded_file = st.file_uploader("Choose a Python file", accept_multiple_files=False)
    if uploaded_file:
        for dataclass in DataclassFinder(uploaded_file.read()):
            st.markdown(f"#### {dataclass['name']}")
            st.markdown("---")
            for element in dataclass["elements"]:
                command[element['name']] = st.text_input(f"{element['name']}: ", f"{element['default'] if element['default'] else ''}", help=f"type: {element['type']}\n\n{element['help'] if element['help'] else ''}")
            st.markdown("---")

        run = f"python {uploaded_file.name}"
        for key, value in command.items():
            if value:
                run += f" --{key}={value}"
        st.success(run)

if __name__ == "__main__":
    main()
```
![image](https://github.com/oh-gnues-iohc/oh-gnues-iohc.github.io/assets/79557937/656cf16f-128d-4bc4-9609-9d46cd6ed5ea)

간단하고, 아직 추가할 코드가 많지만 얼추 완성