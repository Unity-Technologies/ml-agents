# 설치

ML-Agents를 설치하고 사용하기 위해 유니티를 설치해야 하고 이 Repository(저장소)를
Clone(복제)하고 추가종속성을 가지는 Python(파이썬)을 설치해야합니다. 아래 Subsection(하위섹션)에서는 Docker(도커) 설정 외에도
각 단계를 개괄적으로 설명합니다.

## **Unity 2018.4** 또는 이후의 버전을 설치하십시오.

[다운로드](https://store.unity.com/kr/download)하고 설치하십시오. 만약 저희의 도커 설정(차후에 소개할)을 사용하고 싶다면,
유니티를 설치할 때, Linux Build Support를 설정하십시오.

<p align="center">
  <img src="images/unity_linux_build_support.png"
       alt="Linux Build Support"
       width="500" border="10" />
</p>

## Windows 사용자
Windows에서 환경을 설정하기 위해, [세부 사항](Installation-Anaconda-Windows.md)에 설정 방법에 대해 작성하였습니다.
Mac과 Linux는 다음 가이드를 확인해주십시오.

## Mac 또는 Unix 사용자

### ML-Agents Toolkit 저장소 복제

유니티 설치 후에 ML-Agents Toolkit 깃허브 저장소를 설치하고 싶을 것입니다.

```sh
git clone https://github.com/Unity-Technologies/ml-agents.git
```

`UnitySDK` 하위 디렉토리에는 프로젝트에 추가할 유니티 애셋이 포함되어 있습니다.
또한 시작하는데 도움이 되는 많은 [예제 환경](Learning-Environment-Examples.md)들이 있습니다.

`ml-agents` 하위 디렉토리에는 유니티 환경과 함게 사용하는 심층 강화학습 트레이너 파이썬 패키지가 포함되어 있습니다.

`ml-agents-envs` 하위 디렉토리에는 `ml-agents` 패키지에 종속되는 유니티의 인터페이스를 위한 파이썬 API가 포함되어 있습니다.

`gym-unity` 하위 디렉토리에는 OpenAI Gym의 인터페이스를 위한 패키지가 포함되어 있습니다.

### 파이썬과 mlagents 패키지 설치

ML-Agents toolkit을 사용하기 위해 [setup.py file](../ml-agents/setup.py)에 나열된 종속성과 함께 파이썬 3.6이 필요합니다.
주요 종속성의 일부는 다음을 포함합니다:

- [TensorFlow](Background-TensorFlow.md) (Requires a CPU w/ AVX support)
- [Jupyter](Background-Jupyter.md)

Python 3.6이 만약 설치되어 있지 않다면, [다운로드](https://www.python.org/downloads/)하고 설치하십시오.

만약 당신의 파이썬 환경이 `pip3`을 포함하지 않는다면, 다음
[지시사항](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)
을 따라서 설치하십시오.

종속성과 `mlagents` 파이썬 패키지를 설치하기 위해 다음 명령어를 실행하십시오:

```sh
pip3 install mlagents
```

이 명령어를 통해 PyPi로 부터(복제된 저장소가 아닌) `ml-agents`가 설치될 것입니다.
만약 성공적으로 설치를 완료 했다면, `mlagents-learn --help` 명령어를 실행할 수 있을 것입니다.
명령어를 실행하면 유니티 로고와 `mlagents-learn`에서 사용할 수 있는 명령어 라인 매개변수들을 볼 수 있습니다.

**주의:**

- 현재 Python 3.7 또는 Python 3.5을 지원하지 않습니다.
- 만약 Anaconda를 사용하고 TensorFlow에 문제가 있다면, 다음
  [링크](https://www.tensorflow.org/install/pip)에서 Anaconda 환경에서 어떻게 TensorFlow를 설치하는지 확인하십시오.
### 개발을 위한 설치방법

만약 `ml-agents` 또는 `ml-agents-envs`를 수정하고 싶다면, PyPi가 아닌 복제된 저장소로 부터 패키지를 설치해야 합니다.
이를 위해, `ml-agents`와 `ml-agents-envs`를 각각 설치해야 합니다. 저장소의 루트 디렉토리에서 다음 명령어를 실행하십시오:

```sh
cd ml-agents-envs
pip3 install -e ./
cd ..
cd ml-agents
pip3 install -e ./
```

`-e` 플래그를 사용하여 pip를 실행 하면 파이썬 파일을 직접 변경할 수 있고 `mlagents-learn`를 실행할 때 반영됩니다.
`mlagents` 패키지가 `mlagents_envs`에 의존적이고, 다른 순서로 설치하면 PyPi로 부터 `mlagents_envs`를
설치할 수 있기 때문에 이 순서대로 패키지를 설치하는 것은 중요합니다.

## 도커 기반 설치

만약 ML-Agents를 위해 도커를 사용하고 싶다면, [이 가이드](Using-Docker.md)를 따라하십시오.

## 다음 단계

[기초 가이드](Basic-Guide.md) 페이지에는 유니티 내에서 ML-Agents toolkit의 설정 및 학습된 모델 실행,
환경 구축, 학습 방법에 대한 여러 짧은 튜토리얼을 포함하고 있습니다.

## 도움말

ML-Agents와 관련된 문제가 발생하면 저희의 [FAQ](FAQ.md)와 [제약 사항](Limitations.md) 페이지를 참고해 주십시오.
만약 문제에 대한 아무것도 찾을 수 없다면 OS, Pythons 버전 및 정확한 오류 메세지와 함께 [이슈 제출](https://github.com/Unity-Technologies/ml-agents/issues)을 해주십시오.


## 한글 번역

해당 문서의 한글 번역은 [장현준 (Hyeonjun Jang)]([https://github.com/janghyeonjun](https://github.com/janghyeonjun))에 의해 진행되었습니다. 내용상 오류나 오탈자가 있는 경우 totok682@naver.com 으로 연락주시면 감사드리겠습니다.
