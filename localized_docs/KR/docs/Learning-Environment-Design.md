# 학습 환경 설계하기

이 페이지에는 Scene 안에서 에이전트를 설계하는 것과 반대로 Scene 및 시뮬레이션을 설정하는 것과 관련된 ML-Agents Unity SDK의 개요와 함께 학습 환경을 설계하는 방법에 대한 일반적인 조언이 포함되어 있습니다. [에이전트 디자인하기](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Design-Agents.md)에는 관찰, 작업 및 보상을 측정하고, 다중 에이전트 시나리오를 위한 팀을 정의하며, 모방 학습을 위한 에이전트 시연을 기록하는 방법을 포함한다. ML-Agents Toolkit에서 제공하는 전체 기능 집합에 대한 온보드 지원을 위해 [API 설명서](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/API-Reference.md)를 탐색하는 것이 좋습니다. 또한 [예제 환경](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Examples.md)은 거의 모든 기능의 샘플을 제공하는 훌륭한 리소스입니다.

## 시뮬레이션 및 교육 프로세스

훈련과 시뮬레이션은 ML-Agents 아카데미 클래스에 의해 조정된 단계로 진행됩니다. 아카데미는 Scene에서 에이전트 객체를 사용하여 시뮬레이션을 진행합니다.

훈련 중, 외부 파이썬 훈련 프로세스는 아카데미와 통신하여 데이터를 수집하고 신경망 모델을 최적화하는 동안 일련의 에피소드를 실행합니다. 훈련이 성공적으로 완료되면, 나중에 사용할 수 있도록 훈련된 모델 파일을 Unity 프로젝트에 추가할 수 있습니다.

ML-Agents 아카데미 클래스는 다음과 같이 에이전트 시뮬레이션 루프를 조정합니다.

1. 아카데미의 `OnEnvironmentReset` 델리게이트를 호출합니다.
1. Scene의 각 에이전트에 대해 `OnEpisodeBegin()` 함수를 호출합니다.
1. Scene에 대한 정보를 수집합니다. 이는 Scene의 각 에이전트에 대해 `CollectObservations(VectorSensor sensor)` 기능을 호출하고, 센서를 업데이트하고 결과 관찰을 수집하는 방식으로 이루어집니다.
1. 각 에이전트의 규칙을 사용하여 에이전트의 다음 작업을 결정합니다.
1. Scene의 각 에이전트에 대해 `OnActionReceived()` 함수를 호출하여 에이전트의 규칙에서 선택한 작업을 전달합니다.
1. 에이전트가 `Max Step` 카운트에 도달했거나 `EndEpisode()`로 표시된 경우 에이전트의 `OnEpisodeBegin()` 함수를 호출합니다.

훈련 환경을 만들려면 에이전트 클래스를 확장하여 특정 시나리오에 따라 위의 메서드를 구현해야 하는지 여부를 결정합니다.

## Unity Scene 체계화하기

Unity Scene에서 ML-Agents Toolkit을 교육하고 사용하려면, Scene은 에이전트 하위 클래스 만큼 필요합니다. 에이전트 인스턴스는 해당 에이전트를 나타내는 GameObject에 연결되어야 합니다.

### 아카데미

아카데미는 에이전트들과 에이전트 끼리의 의사결정 과정을 조정하는 싱글톤입니다. 한 번에 단 하나의 아카데미만 존재합니다.

#### 아카데미 재설정하기

각 에피소드가 시작될 때, 환경을 변경하려면 아카테미의 OnEnvironmentReset 작업에 메서드를 추가합니다.

```csharp
public class MySceneBehavior : MonoBehaviour
{
    public void Awake()
    {
        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }

    void EnvironmentReset()
    {
        // Reset the scene here
    }
}
```

예를 들어, 에이전트를 시작 위치로 재설정하거나 목표를 임의의 위치로 이동할 수 있습니다. 파이썬 `UnityEnvironment`에서 `reset()` 메서드가 호출되면 환경이 재설정됩니다.

환경을 재설정할 때, 훈련을 다른 조건으로 일반화할 수 있도록 변경해야 할 요소를 고려하세요. 예를 들어, 미로 해결 에이전트를 훈련시키고 있다면, 각 훈련 에피소드에 대해 미로 자체를 변경하세요. 그렇지 않으면 에이전트는 일반적인 미로가 아닌 특정 미로만을 해결하는 방법을 배울 것입니다.

### 다중 영역

많은 예제 환경에서 훈련 영역의 많은 복사본이 장면에서 인스턴스화됩니다. 이는 일반적으로 훈련 속도를 높여 환경이 많은 경험을 병렬로 모을 수 있게 합니다. 이 작업은 같은 행동 이름을 가진 많은 에이전트를 인스턴스화하는 것만으로 수행할 수 있습니다. 가능한 경우 여러 영역을 지원하도록 Scene을 설계하는 것이 좋습니다.

예제 환경을 확인하여 여러 영역의 예제를 확인하세요. 또한 [새로운 학습 환경 만들기](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Create-New.md#optional-multiple-training-areas-within-the-same-scene) 가이드에서 이 옵션을 설명합니다.

## 환경

Unity에서 훈련 환경을 만들 때 외부 교육 프로세스에 의해 Scene을 제어할 수 있도록 설정해야 합니다. 고려해야 할 사항은 다음과 같습니다.

- 훈련 프로세스에 의해 Unity 응용 프로그램이 시작되면 훈련 Scene이 자동으로 시작되어야 합니다.
- 아카데미는 각 교육 에피소드에 대해 장면을 유효한 시작 지점으로 재설정해야 합니다.
- 훈련 에피소드는 확실한 끝이 있어야 합니다.
- `Max Steps`를 사용하거나 `EndEpisode()`로 에피소드를 수동으로 끝내세요.

## 환경 파라미터

커리큘럼 학습과 환경 파라미터 랜덤화는 환경의 특정 파라미터를 제어하는 두 가지 훈련 방법입니다. 따라서 각 단계에서 환경 파라미터를 올바른 값으로 업데이트해야 합니다. 이를 가능하게 하기 위해 두 기능 모두에 대한 훈련 구성에 정의된 파라미터의 값을 검색하는 데 사용할 수 있는 `EnvironmentParameters ` C# 클래스를 노출합니다. 자세한 내용은 커리큘럼 학습 및 환경 파라미터 랜덤화에 대한 [설명](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md#environment-parameters)을 참조하세요.

아카데미를 활용하여 에이전트의 `OnEpisodeBegin()` 함수에서 `Academy.Instance.EnvironmentParameters`을 사용하여 환경을 수정하는 것을 추천합니다. 샘플 사용은 WallJump 환경 예제를 참조하세요 ([WallJumpAgent.cs](https://github.com/Unity-Technologies/ml-agents/blob/develop/Project/Assets/ML-Agents/Examples/WallJump/Scripts/WallJumpAgent.cs)

## 에이전트

에이전트 클래스는 관찰을 수집하고 작업을 수행하는 Scene의 배우를 나타냅니다. 에이전트 클래스는 축구 게임의 플레이어 개체 또는 자동차 시뮬레이션의 자동차 개체와 같이 일반적으로 배우를 나타내는 Scene의 게임 개체에 연결됩니다. 모든 에이전트에는 적절한 `행동 파라미터`가 있어야 합니다.

일반적으로, 에이전트를 생성할 때 에이전트 클래스를 확장하고 `CollectObservations(VectorSensor sensor)` 및 `OnActionReceived()` 메서드를 구현해야 합니다.

- `CollectObservations(VectorSensor sensor)` — 환경에 대한 에이전트의 관찰을 수집합니다.
- `OnActionReceived()` — 에이전트의 규칙에서 선택한 작업을 수행하고 현재 상태에 보상을 할당합니다.

이러한 기능의 구현에 따라 이 에이전트에 할당된 동작 파라미터를 설정하는 방법이 결정됩니다.

또한 에이전트가 작업을 완수하거나 시간을 초과하는 방법을 결정해야 합니다. 에이전트가 작업을 완료한 경우 (또는 취소할 수 없는 실패) `EndEpisode()` 함수를 호출하여 `OnActionReceived()` 함수에서 에이전트 에피소드를 수동으로 종료할 수 있습니다. 또한 에이전트의 `Max Steps` 속성을 양수 값으로 설정할 수 있으며 에이전트는 많은 단계를 수행한 후 에피소드를 검토합니다. 에이전트를 다시 시작할 수 있도록 준비하기 위해 `Agent.OnEpisodeBegin()` 함수를 사용할 수 있습니다.

에이전트 프로그래밍에 대한 자세한 내용은 [에이전트](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Design-Agents.md)를 참조하세요.

## 통계 기록하기

개발자들에게 Unity 환경 내에서 통계를 기록할 수 있는 메커니즘을 제공하고 있습니다. 이러한 통계는 훈련 과정 중에 집계되고 생성됩니다. 통계를 기록하려면 `StatsRecorder` C# 클래스를 참조하세요.

샘플 사용 방법을 보시려면 FoodCollector 예제 환경의 [FoodCollectorSettings.cs](https://github.com/Unity-Technologies/ml-agents/blob/develop/Project/Assets/ML-Agents/Examples/FoodCollector/Scripts/FoodCollectorSettings.cs)를 참조하세요.
