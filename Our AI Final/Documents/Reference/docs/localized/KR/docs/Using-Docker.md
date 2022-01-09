# ML-Agents 용 도커 사용법

도커를 사용해 추론과 학습을 하고자하는 Windows와 Mac 사용자를 위한 솔루션을 제공합니다.
이것은 Python과 TensorFlow 설치를 피하고자 하는 분에게 매력적인 옵션이 될 것입니다. 현재 설정은 TensorFlow와 Unity가 _CPU를 통해서만_
계산하도록 합니다. 따라서 도커 시뮬레이션은 GPU를 사용하지 않고 시각적 렌더링을 위해 [`Xvfb`](https://en.wikipedia.org/wiki/Xvfb)를 사용합니다.
`Xvfb`는 `ML-Agents`(또는 다른 응용 프로그램)가 가상으로 렌더링을 할 수 있게하는 유틸리티 입니다. 즉, `ML-Agents`를 실행하는 기계가 GPU를 가지고 있거나
디스플레이를 가지고 있다고 가정하지 않습니다. 이것은 카메라 기반의 시각적 관찰 요소가 포함된 환경은 더욱 느려질 수도 있음을 의미합니다.

## 요구사항

- 유니티 _Linux Build Support_ 컴포넌트
- [도커](https://www.docker.com)

## 설치

- 유니티 인스톨러를 [다운로드](https://unity3d.com/kr/get-unity/download)하고 _Linux Build Support_ 컴포넌트를 추가하십시오.

- 도커가 설치되어 있지 않다면 [다운로드](https://www.docker.com/community-edition#/download)하고 설치 하십시오.

- 호스트 머신과 분리된 환경에서 도커를 실행하기 때문에, 호스트 머신안에 마운트된 디렉토리는 트레이너 환경 설정 파일,
  유니티 실행 파일, 커리큘럼 파일과 TensorFlow 그래프와 같은 데이터를 공유하기위해 사용됩니다.
  이를 위해, 편의상 비어있는 `unity-volume` 디렉토리를 저장소의 루트에 만들었으나, 다른 디렉토리의 사용은 자유롭게 할 수 있습니다.
  이 가이드의 나머지 부분에서는 `unity-volume` 디렉토리가 사용된다고 가정하고 진행됩니다.

## 사용법

ML-Agents 용 도커 사용에는 세 단계가 포함됩니다.: 특정 플래그를 사용하여 유니티 환경 빌드, 도커 컨테이너 빌드
마지막으로, 컨테이너 실행. 만약 ML-Agents 용 유니티 환경 빌드에 익숙하지 않다면, [3D 밸런스 볼 예제와 함께 시작하기](Getting-Started-with-Balance-Ball.md) 가이드를 먼저 읽으십시오.

### 환경 빌드 (옵션)

_학습을 위해 에디터 사용을 원한다면 이 단계를 건너뛸 수 있습니다._

도커는 일반적으로 호스트 머신과 (리눅스) 커널을 공유하는 컨테이너를 실행하기 때문에,
유니티 환경은 리눅스 플랫폼이 구축되어야 합니다. 유니티 환경을 빌드할 때, 빌드 세팅 창(Build Settings window)에서
다음 옵션을 선택해 주십시오:

- 타겟 플랫폼을 `리눅스`로 설정 (Set the _Target Platform_ to `Linux`)
- _아키텍처_를 `x86_64'로 설정 (Set the _Architecture_ to `x86_64`)
- 환경에서 시각적인 관찰을 필요로 하지않는다면, `headless` 옵션을 선택할 수 있습니다 (아래 사진 참조).

`빌드` (Build)를 클릭하고, 환경 이름을 선택하고 (예시: `3DBall`) 출력 디레토리를 `unity-volume`으로 설정하십시오.
빌드 후에, 파일 `<환경 이름>.x86_64` 와 하위디렉토리 `<환경 이름>_Data/` 가 `unity-volume` 에 생성 되어있는지 확인하십시오.

![도커를 위한 빌드 설정](images/docker_build_settings.png)

### 도커 컨테이너 빌드

첫 번째, 도커 머신이 시스템에서 작동하는지 확인하십시오. 저장소의 최상단에서 다음 명령어를 호출하여
도커 컨테이너를 빌드하십시오:

```sh
docker build -t <image-name> .
```

`<image-name>`을 도커 이미지 이름으로 바꾸십시오, 예시: `balance.ball.v0.1`.

### 도커 컨테이너 실행

저장소의 최상단에서 다음 명령어를 호출하여 도커 컨테이너를 실행하십시오:

```sh
docker run --name <container-name> \
           --mount type=bind,source="$(pwd)"/unity-volume,target=/unity-volume \
           -p 5005:5005 \
           <image-name>:latest \
           --docker-target-name=unity-volume \
           <trainer-config-file> \
           --env=<environment-name> \
           --train \
           --run-id=<run-id>
```

인수(argument) 값 정보:

- `<container-name>` 은 컨테이너를 구분하기위해 사용됩니다 (컨테이너를 인터럽트하거나 종료시킬 때).
이것은 선택사항이며 설정하지 않았을 경우 도커는 랜덤한 이름을 생성합니다. _도커 이미지를 실행할 때마다
고유한 이름을 가져야함에 유의하십시오._
- `<image-name>` 컨테이너를 빌드할 때 사용할 image name을 참조합니다.
- `<environment-name>` __(옵션)__: 리눅스 실행파일과 함께 학습을 할 경우, 인수 값이 실행파일의 이름이 된다.
에디터에서 학습을 할 경우, `<environment-name>` 인수를 전달하지 말고 유니티에서 _"Start training by pressing
  the Play button in the Unity Editor"_ 메세지가 화면에 표시될 때 :arrow_forward: 버튼을 누르십시오.
- `source`: 유니티 실행파일을 저장할 호스트 운영체제의 경로를 참조합니다.
- `target`: 도커가`source` 경로에 이 이름을 가진 디스크로 마운트하도록 합니다.
- `docker-target-name`: ML-Agents 파이썬 패키지에게 유니티 실행파일을 읽고 그래프를 저장할 수 있는 디스크의 이름을 알려준다.
**그러므로 `target`과 동일한 값을 가져야 합니다.**
- `trainer-config-file`, `train`, `run-id`: ML-Agents 인자들은 `mlagents-learn`로 전달됩니다. 트레이너 설정 파일의 이름 `trainer-config-file`,
알고리즘을 학습하는 `train`, 그리고 각 실험에 고유한 식별자를 태깅하는데 사용되는 `run-id`.
컨테이너가 파일에 접근할 수 있도록 trainer-config 파일을 `unity-volume` 안에 둘 것을 권장합니다.

`3DBall` 환경 실행파일을 학습하기 위해 다음 명령어가 사용됩니다:

```sh
docker run --name 3DBallContainer.first.trial \
           --mount type=bind,source="$(pwd)"/unity-volume,target=/unity-volume \
           -p 5005:5005 \
           balance.ball.v0.1:latest 3DBall \
           --docker-target-name=unity-volume \
           trainer_config.yaml \
           --env=3DBall
           --train \
           --run-id=3dball_first_trial
```

도커 마운트에 대한 세부 사항은 도커의 [이 문서](https://docs.docker.com/storage/bind-mounts/)를 참고해 주십시오.

**참고** 도커를 사용해 시각적인 관찰을 포함한 환경을 학습할 경우, 콘테이너를 위해 할당한 도커의 디폴트 메모리를 늘려야할 것입니다.
예를 들어, [여기](https://docs.docker.com/docker-for-mac/#advanced) Mac 사용자를 위한 도커 지시사항을 봐주십시오.

### 컨테이너 중지 및 상태 저장

학습 진행 상황에 만족했을 경우, 상태를 저장하는 동안 `Ctrl+C` or `⌘+C` (Mac) 키를 사용하거나 다음 명령어를 통해 도커 컨테이너를 중지할 수 있습니다:

```sh
docker kill --signal=SIGINT <container-name>
```

`<container-name>` 은 `docker run` 명령어에 지정된 컨테이너 이름입니다. 지정하지 않으면 무작위로 생성되며`docker container ls`를 통해 확인할 수 있습니다.


## 한글 번역

해당 문서의 한글 번역은 [장현준 (Hyeonjun Jang)]([https://github.com/janghyeonjun](https://github.com/janghyeonjun))에 의해 진행되었습니다. 내용상 오류나 오탈자가 있는 경우 totok682@naver.com 으로 연락주시면 감사드리겠습니다.
