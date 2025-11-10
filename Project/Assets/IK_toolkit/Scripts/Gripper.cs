using System.Collections;
using UnityEngine;

/// <summary>
/// 그리퍼의 왼쪽, 오른쪽 핑거를 제어하는 스크립트.
/// Open()과 Close() 메서드를 호출하여 그리퍼를 작동시킬 수 있습니다.
/// </summary>
public class Gripper : MonoBehaviour
{
    [Header("Gripper Parts")]
    [Tooltip("그리퍼의 왼쪽 핑거 Transform")]
    public Transform leftFinger;

    [Tooltip("그리퍼의 오른쪽 핑거 Transform")]
    public Transform rightFinger;

    [Header("Gripper Settings")]
    [Tooltip("핑거가 닫혔을 때 각 핑거가 안쪽으로 이동할 거리 (로컬 X축 기준)")]
    public float closedPositionOffset = 0.1f;

    [Tooltip("핑거가 움직이는 속도")]
    public float speed = 2.0f;

    // 각 핑거의 초기 위치와 목표 위치
    public Vector3 initialLeftFingerPos;
    public Vector3 initialRightFingerPos;
    public Vector3 targetLeftFingerPos;
    public Vector3 targetRightFingerPos;

    /// <summary>
    /// 스크립트가 시작될 때 초기 위치를 저장합니다.
    /// </summary>
    void Start()
    {
        // 핑거 오브젝트가 할당되었는지 확인
        if (leftFinger == null || rightFinger == null)
        {
            Debug.LogError("GripperController: 핑거 Transform이 할당되지 않았습니다!");
            return;
        }

        // 각 핑거의 로컬 시작 위치를 저장합니다.
        initialLeftFingerPos = leftFinger.localPosition;
        initialRightFingerPos = rightFinger.localPosition;

        // 초기 상태는 열려있는 상태로 설정
        targetLeftFingerPos = initialLeftFingerPos;
        targetRightFingerPos = initialRightFingerPos;
    }

    /// <summary>
    /// 매 프레임마다 목표 위치로 핑거를 부드럽게 이동시킵니다.
    /// </summary>
    void Update()
    {
        // --- 테스트용 키보드 입력 ---
        // 'G' 키를 누르면 그리퍼를 닫습니다.
        if (Input.GetKeyDown(KeyCode.G))
        {
            Close();
            Debug.Log("Gripper Test: Close (G key pressed)");
        }

        // 'H' 키를 누르면 그리퍼를 엽니다.
        if (Input.GetKeyDown(KeyCode.H))
        {
            Open();
            Debug.Log("Gripper Test: Open (H key pressed)");
        }

        // 핑거가 없으면 실행하지 않음
        if (leftFinger == null || rightFinger == null) return;

        // 왼쪽 핑거를 목표 위치로 이동
        leftFinger.localPosition = Vector3.MoveTowards(leftFinger.localPosition, targetLeftFingerPos, speed * Time.deltaTime);

        // 오른쪽 핑거를 목표 위치로 이동
        rightFinger.localPosition = Vector3.MoveTowards(rightFinger.localPosition, targetRightFingerPos, speed * Time.deltaTime);
    }

    /// <summary>
    /// 그리퍼를 닫습니다. (핑거를 안쪽으로 오므립니다)
    /// </summary>
    public void Close()
    {
        // 왼쪽 핑거의 목표 위치: 초기 위치에서 로컬 X축으로 +offset 만큼 이동
        targetLeftFingerPos = initialLeftFingerPos + new Vector3(closedPositionOffset, 0, 0);

        // 오른쪽 핑거의 목표 위치: 초기 위치에서 로컬 X축으로 -offset 만큼 이동
        targetRightFingerPos = initialRightFingerPos + new Vector3(-closedPositionOffset, 0, 0);
    }

    /// <summary>
    /// 그리퍼를 엽니다. (핑거를 원래 위치로 되돌립니다)
    /// </summary>
    public void Open()
    {
        // 각 핑거의 목표 위치를 저장해둔 초기 위치로 설정
        targetLeftFingerPos = initialLeftFingerPos;
        targetRightFingerPos = initialRightFingerPos;
    }
}
