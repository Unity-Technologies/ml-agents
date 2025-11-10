using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;

public class UR16Agent : Agent
{
    [Tooltip("에이전트가 움직일 IK 타겟입니다.")]
    public Transform ikTarget;

    [Tooltip("에이전트가 따라가야 할 목표 물체입니다.")]
    public Transform targetObject;

    [Tooltip("IK 타겟의 이동 속도입니다.")]
    public float moveSpeed = 1.0f;

    public TouchSensor sensor;

    private Vector3 initialIKTargetPosition;

    // 시작시 한번만 불러지는 초기화 함수
    // 로봇의 엔드이펙터가 초기 위치로
    public override void Initialize()
    {
        initialIKTargetPosition = ikTarget.localPosition;
    }

    //  각 학습 에피소드가 시작될 때 호출
    //  이 메서드에서는 로봇의 위치, 컨베이어 벨트 위의 물체 위치 및 종류 등을 초기화하는 코드를 작성
    public override void OnEpisodeBegin()
    {
        // IK 타겟의 위치를 초기 상태로 리셋합니다.
        ikTarget.localPosition = initialIKTargetPosition;

        // 목표 물체의 위치를 무작위로 변경하여 일반화 능력을 학습시킵니다.
        // 이 예제에서는 특정 범위 내에서 무작위 위치를 설정합니다.
        // 실제 환경에 맞게 범위를 조절하세요.
        targetObject.localPosition = new Vector3(Random.Range(-0.66f, 0.66f),
                                                 0.5f,
                                                 Random.Range(0.7f, 1.97f));
    }

    // 에이전트가 주변 환경을 관찰하고 상태(State)를 수집하는 부분
    // 로봇의 각 관절 각도, 그리퍼의 상태, 목표 물체의 위치, 카메라를 통해 얻은 시각적 정보 등을 관측 값으로 추가
    // 이 예제에서는 CameraSensorComponent를 사용하므로 시각적 관찰은 자동으로 처리됩니다.
    // 추가적인 벡터 관찰(예: IK 타겟의 위치)을 여기에 추가할 수 있습니다.
    public override void CollectObservations(VectorSensor sensor)
    {
        // IK 타겟의 현재 위치를 관찰 값으로 추가합니다.
        // 이를 통해 에이전트는 자신의 '손' 위치를 인지할 수 있습니다.
        sensor.AddObservation(ikTarget.localPosition);

        // 목표 물체의 위치도 관찰 값으로 추가하면 학습이 더 빨라질 수 있습니다.
        sensor.AddObservation(targetObject.localPosition);
    }

    // 정책(Policy)으로부터 받은 행동(Action)을 실행하는 부분입니다.
    // IK 타겟의 위치를 조정하거나 그리퍼를 열고 닫는 등의 행동을 정의합니다.
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 연속 행동(Continuous Actions) 값을 받아옵니다.
        // 값은 3개(x, y, z)이며, 각각 -1과 1 사이의 값을 가집니다.
        float moveX = actionBuffers.ContinuousActions[0];
        float moveY = actionBuffers.ContinuousActions[1];
        float moveZ = actionBuffers.ContinuousActions[2];

        // 행동 값을 IK 타겟의 이동 방향으로 변환하고, 속도와 시간에 맞춰 움직입니다.
        ikTarget.localPosition += new Vector3(moveX, moveY, moveZ) * Time.deltaTime * moveSpeed;

        // --- 보상(Reward) 설계 ---

        // IK 타겟과 목표 물체 사이의 거리를 계산합니다.
        float distanceToTarget = Vector3.Distance(ikTarget.position, targetObject.position);

        // 목표에 충분히 가까워지면 긍정적 보상을 주고 에피소드를 종료합니다.
        // 시도2: 0.05f, 보상 1
        // 시도3: 0.2f, 보상 2
        // 시도4: 0.2f, 보상 10점
        if (distanceToTarget < 0.05f) // 이 값은 씬의 스케일에 맞게 조절해야 합니다.
        {
            SetReward(10.0f);
            EndEpisode();
        }

        float distanceToOrigin = Vector3.Distance(ikTarget.position, transform.position);
        // 시도1: 너무 멀어지거나 가까워지면 벌점을 주고 초기화
        if(distanceToOrigin > 2.1f || distanceToOrigin < 0.4f)
        {
            AddReward(-0.1f);
            EndEpisode();
        }

        if(sensor.isContacted)
        {
            sensor.isContacted = false;
            AddReward(-0.1f);
            EndEpisode();
        }


        // 목표에서 멀어지면 작은 부정적 보상을 줍니다.
        // 이는 에이전트가 목표 주변을 맴돌지 않고 빠르게 도달하도록 유도합니다.
        AddReward(-0.003f);
    }

    // (선택 사항) 사람이 직접 로봇을 조작하여 학습 과정을 테스트해볼 수 있는 메서드입니다.
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions.Clear();

        // 키보드 입력을 받아 IK 타겟을 직접 조종합니다.
        // W/S: Z축, A/D: X축, Q/E: Y축
        continuousActions[0] = Input.GetAxis("Horizontal"); // D, A 키
        continuousActions[1] = Input.GetKey(KeyCode.E) ? 1.0f : (Input.GetKey(KeyCode.Q) ? -1.0f : 0f); // E, Q 키
        continuousActions[2] = Input.GetAxis("Vertical");   // W, S 키
    }
}
