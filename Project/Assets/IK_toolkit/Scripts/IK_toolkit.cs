using System;
using System.Collections.Generic;
using UnityEngine;

// IK_toolkit: UR16e 로봇 팔의 역기구학 툴킷
[ExecuteInEditMode]
public class IK_toolkit : MonoBehaviour
{
    public Transform ik; // IK 계산을 위한 Transform
    public int solutionID; // 선택된 솔루션의 ID
    [SerializeField]private List<string> IK_Solutions = new List<string>(); // 가능한 IK 솔루션 목록
    public List<double> goodSolution = new List<double>(); // 유효한 솔루션 저장
    public List<Transform> robot = new List<Transform>(); // 로봇 관절 Transform 목록

    // UR16e 로봇 팔의 Denavit-Hartenberg 파라미터 행렬
    public static double[,] DH_matrix_UR16e = new double[6, 3] {
        { 0, Mathf.PI / 2.0, 0.1807 },
        { -0.4784, 0, 0 },
        { -0.36, 0, 0 },
        { 0, Mathf.PI / 2.0, 0.17415 },
        { 0, -Mathf.PI / 2.0, 0.11985},
        { 0, 0, 0.11655}
    };

    // 매 프레임마다 호출되는 Update 메서드
    void Update()
    {
        // IK를 위한 변환 행렬 계산
        Matrix4x4 transform_matrix = GetTransformMatrix(ik);

        // Y축을 기준으로 행렬을 반사
        Matrix4x4 mt = Matrix4x4.identity;
        mt.m11 = -1; // Y축 반사 행렬 설정
        Matrix4x4 mt_inverse = mt.inverse; // 반사 행렬의 역행렬
        Matrix4x4 result = mt * transform_matrix * mt_inverse; // 최종 변환 행렬

        // 역기구학 솔루션 계산
        double[,] solutions = Inverse_kinematic_solutions(result);
        IK_Solutions.Clear(); // 기존 솔루션 목록 초기화
        IK_Solutions = DisplaySolutions(solutions); // 새로운 솔루션 목록 표시

        // 선택된 솔루션에 따라 로봇 팔 관절 설정
        ApplyJointSolution(IK_Solutions, solutions, solutionID, robot);
        goodSolution.Clear(); // 유효한 솔루션 초기화
        // 유효한 솔루션의 각 관절 각도 저장
        for (int i = 0; i < 6; i++) {
            goodSolution.Add(solutions[i, 5]);
        }
    }

    // 주어진 Transform에 대한 변환 행렬을 얻는 메서드
    public static Matrix4x4 GetTransformMatrix(Transform controller)
    {
        // 위치 및 회전을 기반으로 변환 행렬 생성
        return Matrix4x4.TRS(new Vector3(controller.localPosition.x, controller.localPosition.y, controller.localPosition.z), 
                              Quaternion.Euler(controller.localEulerAngles.x, controller.localEulerAngles.y, controller.localEulerAngles.z), 
                              new Vector3(1, 1, 1));
    }

    // Denavit-Hartenberg 파라미터를 사용하여 변환 행렬 계산
    public static Matrix4x4 ComputeTransformMatrix(int jointIndex, double[,] jointAngles)
    {
        jointIndex--;

        // Z축 주위 회전
        var rotationZ = Matrix4x4.identity;
        rotationZ.m00 = Mathf.Cos((float)jointAngles[0, jointIndex]);
        rotationZ.m01 = -Mathf.Sin((float)jointAngles[0, jointIndex]);
        rotationZ.m10 = Mathf.Sin((float)jointAngles[0, jointIndex]);
        rotationZ.m11 = Mathf.Cos((float)jointAngles[0, jointIndex]);

        // Z축을 따라 이동
        var translationZ = Matrix4x4.identity;
        translationZ.m23 = (float)DH_matrix_UR16e[jointIndex, 2];

        // X축을 따라 이동
        var translationX = Matrix4x4.identity;
        translationX.m03 = (float)DH_matrix_UR16e[jointIndex, 0];

        // X축 주위 회전
        var rotationX = Matrix4x4.identity;
        rotationX.m11 = Mathf.Cos((float)DH_matrix_UR16e[jointIndex, 1]);
        rotationX.m12 = -Mathf.Sin((float)DH_matrix_UR16e[jointIndex, 1]);
        rotationX.m21 = Mathf.Sin((float)DH_matrix_UR16e[jointIndex, 1]);
        rotationX.m22 = Mathf.Cos((float)DH_matrix_UR16e[jointIndex, 1]);

        // 변환을 결합: rotationZ, translationZ, translationX, rotationX
        return rotationZ * translationZ * translationX * rotationX;
    }

    // 역기구학 솔루션을 로봇 팔 관절에 적용하는 메서드
    public static void ApplyJointSolution(List<string> solutionStatus, double[,] jointSolutions, int solutionIndex, List<Transform> robotJoints)
    {
        // 솔루션이 유효한지 확인
        if (solutionStatus[solutionIndex] != "NON DISPONIBLE")
        {
            // 로봇의 각 관절에 대해 각도 적용
            for (int i = 0; i < robotJoints.Count; i++)
            {
                robotJoints[i].localEulerAngles = ConvertJointAngles(jointSolutions[i, solutionIndex], i);
            }
        }
        else
        {
            // 솔루션이 없으면 에러 메시지 출력
            Debug.LogError("NO SOLUTION");
        }
    }

    // 관절 각도를 라디안에서 도로 변환하고 오프셋을 적용하는 메서드
    private static Vector3 ConvertJointAngles(double angleRad, int jointIndex)
    {
        float angleDeg = -(float)(Mathf.Rad2Deg * angleRad); // 라디안을 도로 변환

        // 각 관절에 대한 오프셋 적용
        switch (jointIndex)
        {
            case 1:
            case 4:
                return new Vector3(-90, 0, angleDeg);
            case 5:
                return new Vector3(90, 0, angleDeg);
            default:
                return new Vector3(0, 0, angleDeg);
        }
    }

    // 역기구학 솔루션 계산
    public static double[,] Inverse_kinematic_solutions(Matrix4x4 transform_matrix_unity)
    {
        double[,] theta = new double[6, 8]; // 솔루션 각도 배열

        // P05 위치 계산
        Vector4 P05 = transform_matrix_unity * new Vector4()
        {
            x = 0,
            y = 0,
            z = -(float)DH_matrix_UR16e[5, 2],
            w = 1
        };
        
        // 각도 계산
        float psi = Mathf.Atan2(P05[1], P05[0]);
        float phi = Mathf.Acos((float)((DH_matrix_UR16e[1, 2] + DH_matrix_UR16e[3, 2] + DH_matrix_UR16e[2, 2]) / Mathf.Sqrt(Mathf.Pow(P05[0], 2) + Mathf.Pow(P05[1], 2))));

        // 첫 번째 관절 각도 설정
        theta[0, 0] = psi + phi + Mathf.PI / 2;
        // 나머지 각도도 비슷한 방식으로 설정
        for (int i = 1; i <= 7; i++)
        {
            theta[0, i] = psi - phi + Mathf.PI / 2; // 반복문으로 처리
        }

        // 각 관절의 위치를 기반으로 추가 각도 계산
        for (int i = 0; i < 8; i += 4)
        {
            double t5 = (transform_matrix_unity[0, 3] * Mathf.Sin((float)theta[0, i]) - transform_matrix_unity[1, 3] * Mathf.Cos((float)theta[0, i]) - (DH_matrix_UR16e[1, 2] + DH_matrix_UR16e[3, 2] + DH_matrix_UR16e[2, 2])) / DH_matrix_UR16e[5, 2];
            float th5;
            if (1 >= t5 && t5 >= -1)
            {
                th5 = Mathf.Acos((float)t5); // 유효한 경우 아크코사인 계산
            }
            else
            {
                th5 = 0; // 유효하지 않으면 0으로 설정
            }

            // 각도 저장
            if (i == 0)
            {
                theta[4, 0] = th5;
                theta[4, 1] = th5;
                theta[4, 2] = -th5;
                theta[4, 3] = -th5;
                }
            else
            {
                // theta 배열의 특정 인덱스에 th5 값을 할당
                theta[4, 4] = th5; // 4번째 인덱스에 th5 할당
                theta[4, 5] = th5; // 5번째 인덱스에 th5 할당
                theta[4, 6] = -th5; // 6번째 인덱스에 -th5 할당
                theta[4, 7] = -th5; // 7번째 인덱스에 -th5 할당
            }
        }

        // transform_matrix_unity의 역행렬을 계산
        Matrix4x4 tmu_inverse = transform_matrix_unity.inverse;

        // theta 배열의 각 인덱스에 대해 아크탄젠트를 계산
        float th0 = Mathf.Atan2((-tmu_inverse[1, 0] * Mathf.Sin((float)theta[0, 0]) + tmu_inverse[1, 1] * Mathf.Cos((float)theta[0, 0])), 
                                (tmu_inverse[0, 0] * Mathf.Sin((float)theta[0, 0]) - tmu_inverse[0, 1] * Mathf.Cos((float)theta[0, 0])));
        
        float th2 = Mathf.Atan2((-tmu_inverse[1, 0] * Mathf.Sin((float)theta[0, 2]) + tmu_inverse[1, 1] * Mathf.Cos((float)theta[0, 2])), 
                                (tmu_inverse[0, 0] * Mathf.Sin((float)theta[0, 2]) - tmu_inverse[0, 1] * Mathf.Cos((float)theta[0, 2])));
        
        float th4 = Mathf.Atan2((-tmu_inverse[1, 0] * Mathf.Sin((float)theta[0, 4]) + tmu_inverse[1, 1] * Mathf.Cos((float)theta[0, 4])), 
                                (tmu_inverse[0, 0] * Mathf.Sin((float)theta[0, 4]) - tmu_inverse[0, 1] * Mathf.Cos((float)theta[0, 4])));
        
        float th6 = Mathf.Atan2((-tmu_inverse[1, 0] * Mathf.Sin((float)theta[0, 6]) + tmu_inverse[1, 1] * Mathf.Cos((float)theta[0, 6])), 
                                (tmu_inverse[0, 0] * Mathf.Sin((float)theta[0, 6]) - tmu_inverse[0, 1] * Mathf.Cos((float)theta[0, 6])));

        // theta 배열에 아크탄젠트 결과를 저장
        theta[5, 0] = th0;
        theta[5, 1] = th0;
        theta[5, 2] = th2;
        theta[5, 3] = th2;
        theta[5, 4] = th4;
        theta[5, 5] = th4;
        theta[5, 6] = th6;
        theta[5, 7] = th6;

        // 각 솔루션에 대해 반복
        for (int i = 0; i <= 7; i += 2)
        {
            double[,] t1 = new double[1, 6];
            // theta 배열의 값을 t1 배열에 저장
            t1[0, 0] = theta[0, i];
            t1[0, 1] = theta[1, i];
            t1[0, 2] = theta[2, i];
            t1[0, 3] = theta[3, i];
            t1[0, 4] = theta[4, i];
            t1[0, 5] = theta[5, i];

            // 변환 행렬을 계산
            Matrix4x4 T01 = ComputeTransformMatrix(1, t1);
            Matrix4x4 T45 = ComputeTransformMatrix(5, t1);
            Matrix4x4 T56 = ComputeTransformMatrix(6, t1);
            Matrix4x4 T14 = T01.inverse * transform_matrix_unity * (T45 * T56).inverse;

            // P13 벡터 계산
            Vector4 P13 = T14 * new Vector4()
            {
                x = 0,
                y = (float)-DH_matrix_UR16e[3, 2], // DH 매트릭스의 특정 값 사용
                z = 0,
                w = 1
            };

            // theta[2, i]에 대한 th3 계산
            double t3 = (Mathf.Pow(P13[0], 2) + Mathf.Pow(P13[1], 2) - Mathf.Pow((float)DH_matrix_UR16e[1, 0], 2) - 
                         Mathf.Pow((float)DH_matrix_UR16e[2, 0], 2)) / 
                         (2 * DH_matrix_UR16e[1, 0] * DH_matrix_UR16e[2, 0]);
            double th3;
            // t3가 유효 범위 내에 있을 경우 아크코사인을 계산
            if (1 >= t3 && t3 >= -1)
            {
                th3 = Mathf.Acos((float)t3);
            }
            else
            {
                th3 = 0; // t3가 범위를 벗어나면 0으로 설정
            }
            theta[2, i] = th3; // theta 배열에 th3 저장
            theta[2, i + 1] = -th3; // 대칭값 저장
        }

        // 모든 솔루션에 대해 반복
        for (int i = 0; i < 8; i++)
        {
            double[,] t1 = new double[1, 6];
            // theta 배열의 값을 t1 배열에 저장
            t1[0, 0] = theta[0, i];
            t1[0, 1] = theta[1, i];
            t1[0, 2] = theta[2, i];
            t1[0, 3] = theta[3, i];
            t1[0, 4] = theta[4, i];
            t1[0, 5] = theta[5, i];

            // 변환 행렬을 계산
            Matrix4x4 T01 = ComputeTransformMatrix(1, t1);
            Matrix4x4 T45 = ComputeTransformMatrix(5, t1);
            Matrix4x4 T56 = ComputeTransformMatrix(6, t1);
            Matrix4x4 T14 = T01.inverse * transform_matrix_unity * (T45 * T56).inverse;

            // P13 벡터 계산
            Vector4 P13 = T14 * new Vector4()
            {
                x = 0,
                y = (float)-DH_matrix_UR16e[3, 2],
                z = 0,
                w = 1
            };

            // theta[1, i] 계산
            theta[1, i] = Mathf.Atan2(-P13[1], -P13[0]) - 
                           Mathf.Asin((float)(-DH_matrix_UR16e[2, 0] * Mathf.Sin((float)theta[2, i]) / 
                           Mathf.Sqrt(Mathf.Pow(P13[0], 2) + Mathf.Pow(P13[1], 2))));

            double[,] t2 = new double[1, 6];
            // theta 배열의 값을 t2 배열에 저장
            t2[0, 0] = theta[0, i];
            t2[0, 1] = theta[1, i];
            t2[0, 2] = theta[2, i];
            t2[0, 3] = theta[3, i];
            t2[0, 4] = theta[4, i];
            t2[0, 5] = theta[5, i];

            // 변환 행렬을 계산
            Matrix4x4 T32 = ComputeTransformMatrix(3, t2).inverse;
            Matrix4x4 T21 = ComputeTransformMatrix(2, t2).inverse;
            Matrix4x4 T34 = T32 * T21 * T14;

            // theta[3, i] 계산
            theta[3, i] = Mathf.Atan2(T34[1, 0], T34[0, 0]);
        }

        return theta; // 최종 theta 배열 반환
    }

    // 솔루션을 표시하기 위한 메서드
    public static List<string> DisplaySolutions(double[,] solutions)
    {
        List<string> info = new List<string>();

        // 8개의 가능한 솔루션을 반복
        for (int column = 0; column < 8; column++)
        {
            // 모든 조인트 각도가 유효한지 확인
            bool isValidSolution = true;
			for (int row = 0; row < 6; row++)
			{
				// 각 솔루션의 행(row)에 대해 NaN 여부 확인
				if (double.IsNaN(solutions[row, column]))
				{
				    isValidSolution = false; // NaN이 발견되면 유효하지 않은 솔루션으로 설정
				    break; // 반복문 종료
				}
			}
				
			// 솔루션이 유효한 경우, 관절 각도를 포맷하여 info 리스트에 추가
			if (isValidSolution)
			{
				string solutionInfo = ""; // 솔루션 정보를 저장할 문자열 초기화
				for (int row = 0; row < 6; row++)
				{
				    // 라디안 값을 도(degree) 단위로 변환하고 소수점 둘째 자리까지 반올림
				    double angleInDegrees = Math.Round(Mathf.Rad2Deg * solutions[row, column], 2);
				    solutionInfo += $"{angleInDegrees}"; // 각도를 문자열에 추가
				
				    // 마지막 행이 아닐 경우 구분자 추가
				    if (row < 5)
				    {
				        solutionInfo += " | ";
				    }
				}
				info.Add(solutionInfo); // 완성된 솔루션 정보를 info 리스트에 추가
			}
			// 솔루션이 유효하지 않은 경우 "NON DISPONIBLE"을 info 리스트에 추가
			else
			{
				info.Add("NON DISPONIBLE"); // 유효하지 않은 솔루션에 대한 메시지 추가
			}
		}
				
		return info; // info 리스트 반환
	}
}
