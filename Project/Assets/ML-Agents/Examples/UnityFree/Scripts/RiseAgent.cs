using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public enum THRayType
{
    Nothing = 0,
    Goal = 1,
    Wall = 2,
    Enemy = 3
}

public class THRayClass
{
    public THRayType ray_type = THRayType.Nothing;
    public float dist = 100f; // Ray 탐지 거리는 100으로 초기화.
}

public class RiseAgent : Agent
{
    private ShipEscapeEnvController m_GameController;
    float penalty = -0.0003f;
    public TeamType teamType;
    public float maxHealth = 300f;
    [NonSerialized] public float curHealth = 300f;

    public BufferSensorComponent teamBuffer;

    Vector3 start_pos;
    int start_rot_y;
    int rot_y;
    float normalFactor = Mathf.Sqrt(400f * 400f * 2f);
    bool isInitialized = false;

    bool can_forward = true;

    //Queue<Vector2> sub_goals = new Queue<Vector2>();

    const float kEpsilon = 0.00001F;

    Vector2 RotateVector2D(float angle, Vector2 pos)
    {
        double convert = Math.PI * angle / 180.0;
        //float[] vectorUp = { 0f, 1f }; // 각도 0일때 y쪽을 보고있음(2D 기준, 3D 면 z 축)
        //float x = (float)(Math.Cos(convert) * vectorUp[0] - Math.Sin(convert) * vectorUp[1]);
        //float y = (float)(Math.Sin(convert) * vectorUp[0] + Math.Cos(convert) * vectorUp[1]);

        float x = (float)(Math.Cos(convert) * pos.x - Math.Sin(convert) * pos.y);
        float y = (float)(Math.Sin(convert) * pos.x + Math.Cos(convert) * pos.y);

        return new Vector2(Math.Abs(x) > kEpsilon ? x : 0f, Math.Abs(y) > kEpsilon ? y : 0f);
    }

    public void Init()
    {
        can_forward = true;

        if (isInitialized == false)
        {
            start_pos = transform.position;

            start_rot_y = (int)transform.eulerAngles.y;
            isInitialized = true;
            m_GameController = GetComponentInParent<ShipEscapeEnvController>();
        }

        //sub_goals.Enqueue(new Vector2());
        //sub_goals.Enqueue(new Vector2());
        //sub_goals.Enqueue(new Vector2());

        transform.position = start_pos;
        rot_y = start_rot_y;
        //rot_y = start_rot_y = UnityEngine.Random.Range(180, 270);
        transform.eulerAngles = new Vector3(0f, rot_y, 0f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector2 my_pos_2d = new Vector2(transform.localPosition.x, transform.localPosition.z);

        if (teamType == TeamType.red)
        {
            sensor.AddObservation(curHealth / maxHealth); // health normalize
            foreach (Transform goalTR in m_GameController.goalTRs)
            {
                Vector3 goal_local_pos = goalTR.localPosition;
                Vector2 relative_pos = RotateVector2D(rot_y, new Vector2(goal_local_pos.x - my_pos_2d.x, goal_local_pos.z - my_pos_2d.y));
                relative_pos.x = (float)Math.Round(relative_pos.x / normalFactor, 3);
                relative_pos.y = (float)Math.Round(relative_pos.y / normalFactor, 3);
                sensor.AddObservation(relative_pos);
            }
        }
        else
        {
            Vector2 relative_pos = RotateVector2D(rot_y, new Vector2(start_pos.x - my_pos_2d.x, start_pos.z - my_pos_2d.y));
            relative_pos.x = (float)Math.Round(relative_pos.x / normalFactor, 3);
            relative_pos.y = (float)Math.Round(relative_pos.y / normalFactor, 3);
            sensor.AddObservation(relative_pos);
        }

        Vector2 forward_2d = RotateVector2D(-rot_y, Vector2.up);
        forward_2d.x = (float)Math.Round(forward_2d.x, 3);
        forward_2d.y = (float)Math.Round(forward_2d.y, 3);
        sensor.AddObservation(forward_2d);

        // 팀 버퍼 채워주기.
        var red_positions = m_GameController.reds.Where(o => o.gameObject.activeInHierarchy == true && o != this).Select(o => o.transform.position); // 살아 있는 애만
        var blue_positions = m_GameController.blues.Where(o => o != this).Select(o => o.transform.position);

        foreach (Vector3 team_pos in teamType == TeamType.red ? red_positions : blue_positions) // 아군에 대한 정보
        {
            Vector2 relative_pos = RotateVector2D(rot_y, new Vector2(team_pos.x - my_pos_2d.x, team_pos.z - my_pos_2d.y));
            relative_pos.x = (float)Math.Round(relative_pos.x / normalFactor, 3);
            relative_pos.y = (float)Math.Round(relative_pos.y / normalFactor, 3);
            teamBuffer.AppendObservation(new[] { relative_pos.x, relative_pos.y });
        }

        // Static 시 : 0, 1, 2 사용.   Dynamic 시 : 0, 3 사용.. 거리는 주지만 nothing 과 wall 을 굳이 구별하지 않는다. (Dodge-env 따라함)
        // 아마 Observation 한칸을 줄이는 이득이 있지 않을까 싶다..
        THRayClass[] static_rays = new THRayClass[13];
        THRayClass[] dynamic_rays = new THRayClass[13];

        int ray_class_index = 0;
        for (int i = -60; i <= 60; i += 10)
        {
            static_rays[ray_class_index] = new THRayClass();
            dynamic_rays[ray_class_index] = new THRayClass();
            Vector2 rotatedVector = my_pos_2d + RotateVector2D(-rot_y + i, Vector2.up) * 100f;
            // 공통으로 쓰는 정보
            foreach (Vector3 enemy_pos in teamType == TeamType.red ? blue_positions : red_positions) // 적군에 대한 정보
            {
                Vector2 enemy_pos_2d = new Vector2(enemy_pos.x, enemy_pos.z);
                Vector2[,] cross_enemy_lines = { { enemy_pos_2d + Vector2.up, enemy_pos_2d + Vector2.down }, { enemy_pos_2d + Vector2.right, enemy_pos_2d + Vector2.left } };
                for (int j = 0; j < 2; ++j)
                {
                    if (Vector3.Distance(transform.position, enemy_pos) < 200f)
                    {
                        double[] _cross_point = GetCrossPoint(
                            new[] { (double)my_pos_2d.x, my_pos_2d.y },
                            new[] { (double)rotatedVector.x, rotatedVector.y },
                            new[] { (double)cross_enemy_lines[j, 0].x, cross_enemy_lines[j, 0].y },
                            new[] { (double)cross_enemy_lines[j, 1].x, cross_enemy_lines[j, 1].y }
                            );
                        if (_cross_point != null)
                        {
                            float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                            if (tmp_dist < dynamic_rays[ray_class_index].dist)
                            {
                                dynamic_rays[ray_class_index].ray_type = THRayType.Enemy;
                                dynamic_rays[ray_class_index].dist = tmp_dist;
                            }
                        }
                    }
                }
            }

            foreach (Vector2[] linePoint in m_GameController.linePoints2) // 아래쪽 섬(NLL 아래쪽)
            {
                if (Vector2.Distance(my_pos_2d, linePoint[0]) < 200f || Vector2.Distance(my_pos_2d, linePoint[1]) < 200f)
                {
                    double[] _cross_point = GetCrossPoint(
                        new[] { (double)my_pos_2d.x, my_pos_2d.y },
                        new[] { (double)rotatedVector.x, rotatedVector.y },
                        new[] { (double)linePoint[0].x, linePoint[0].y },
                        new[] { (double)linePoint[1].x, linePoint[1].y }
                        );
                    if (_cross_point != null)
                    {
                        float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                        if (tmp_dist < static_rays[ray_class_index].dist)
                        {
                            static_rays[ray_class_index].ray_type = THRayType.Wall;
                            static_rays[ray_class_index].dist = tmp_dist;
                        }
                        if (tmp_dist < dynamic_rays[ray_class_index].dist)
                        {
                            dynamic_rays[ray_class_index].ray_type = THRayType.Nothing;
                            dynamic_rays[ray_class_index].dist = tmp_dist;
                        }
                    }
                }
            }

            if (teamType == TeamType.red) // 레드만 필요한 정보
            {
                foreach (Vector2[] linePoint in m_GameController.goal_line_points)
                {
                    if (Vector2.Distance(my_pos_2d, linePoint[0]) < 200f || Vector2.Distance(my_pos_2d, linePoint[1]) < 200f)
                    {
                        double[] _cross_point = GetCrossPoint(
                            new[] { (double)my_pos_2d.x, my_pos_2d.y },
                            new[] { (double)rotatedVector.x, rotatedVector.y },
                            new[] { (double)linePoint[0].x, linePoint[0].y },
                            new[] { (double)linePoint[1].x, linePoint[1].y }
                            );
                        if (_cross_point != null)
                        {
                            float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                            if (tmp_dist < static_rays[ray_class_index].dist)
                            {
                                static_rays[ray_class_index].ray_type = THRayType.Goal; // 골에 대한 정보.
                                static_rays[ray_class_index].dist = tmp_dist;
                            }
                        }
                    }
                }

                foreach (Vector2[] linePoint in m_GameController.linePoints1) // 위쪽 섬(NLL 위쪽)
                {
                    if (Vector2.Distance(my_pos_2d, linePoint[0]) < 200f || Vector2.Distance(my_pos_2d, linePoint[1]) < 200f)
                    {
                        double[] _cross_point = GetCrossPoint(
                            new[] { (double)my_pos_2d.x, my_pos_2d.y },
                            new[] { (double)rotatedVector.x, rotatedVector.y },
                            new[] { (double)linePoint[0].x, linePoint[0].y },
                            new[] { (double)linePoint[1].x, linePoint[1].y }
                            );
                        if (_cross_point != null)
                        {
                            float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                            if (tmp_dist < static_rays[ray_class_index].dist)
                            {
                                static_rays[ray_class_index].ray_type = THRayType.Wall; // static 에서는 wall
                                static_rays[ray_class_index].dist = tmp_dist;
                            }
                            if (tmp_dist < dynamic_rays[ray_class_index].dist)
                            {
                                dynamic_rays[ray_class_index].ray_type = THRayType.Nothing; // dynamic 에서는 nothing 이지만 거리는 줌.
                                dynamic_rays[ray_class_index].dist = tmp_dist;
                            }
                        }
                    }
                }

                foreach (Vector2[] linePoint in m_GameController.boundary_points1)
                {
                    double[] _cross_point = GetCrossPoint(
                        new[] { (double)my_pos_2d.x, my_pos_2d.y },
                        new[] { (double)rotatedVector.x, rotatedVector.y },
                        new[] { (double)linePoint[0].x, linePoint[0].y },
                        new[] { (double)linePoint[1].x, linePoint[1].y }
                        );
                    if (_cross_point != null)
                    {
                        float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                        if (tmp_dist < static_rays[ray_class_index].dist)
                        {
                            static_rays[ray_class_index].ray_type = THRayType.Wall; // static 에서는 wall
                            static_rays[ray_class_index].dist = tmp_dist;
                        }
                        if (tmp_dist < dynamic_rays[ray_class_index].dist)
                        {
                            dynamic_rays[ray_class_index].ray_type = THRayType.Nothing; // dynamic 에서는 nothing 이지만 거리는 줌.
                            dynamic_rays[ray_class_index].dist = tmp_dist;
                        }
                    }
                }
            }
            else // Blue 만 쓰는 정보
            {
                foreach (Vector2[] linePoint in m_GameController.boundary_points2) // NLL 까지는 아닌데 대충 반으로 나눈 Boundary
                {
                    double[] _cross_point = GetCrossPoint(
                        new[] { (double)my_pos_2d.x, my_pos_2d.y },
                        new[] { (double)rotatedVector.x, rotatedVector.y },
                        new[] { (double)linePoint[0].x, linePoint[0].y },
                        new[] { (double)linePoint[1].x, linePoint[1].y }
                        );
                    if (_cross_point != null)
                    {
                        float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                        if (tmp_dist < static_rays[ray_class_index].dist)
                        {
                            static_rays[ray_class_index].ray_type = THRayType.Wall; // static 에서는 wall
                            static_rays[ray_class_index].dist = tmp_dist;
                        }
                        if (tmp_dist < dynamic_rays[ray_class_index].dist)
                        {
                            dynamic_rays[ray_class_index].ray_type = THRayType.Nothing; // dynamic 에서는 nothing 이지만 거리는 줌.
                            dynamic_rays[ray_class_index].dist = tmp_dist;
                        }
                    }
                }
            }
            ++ray_class_index;
        }

        foreach (THRayClass th_ray in dynamic_rays)
        {
            sensor.AddObservation(th_ray.ray_type == THRayType.Enemy); // 적
            sensor.AddObservation(th_ray.dist / 100f);
        }

        can_forward = true;
        foreach (THRayClass th_ray in static_rays)
        {
            if (th_ray.ray_type == THRayType.Wall) // 벽 일때
            {
                if (th_ray.dist < 2f) // 하나라도 정면에 걸리면 못감.
                {
                    can_forward = false;
                }
            }

            if (teamType == TeamType.red) // 청군은 골을 모르네
            {
                sensor.AddObservation(th_ray.ray_type == THRayType.Goal); // 골
            }
            sensor.AddObservation(th_ray.ray_type == THRayType.Wall); // 벽
            sensor.AddObservation(th_ray.dist / 100f);
        }
    }

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
    public void MoveAgent(ActionSegment<int> act)
    {
        switch (act[1])
        {
            case 1:
                ++rot_y;
                break;
            case 2:
                --rot_y;
                break;
        }

        if (rot_y > 360)
        {
            rot_y -= 360;
        }

        if (rot_y < 0)
        {
            rot_y += 360;
        }

        transform.eulerAngles = new Vector3(0f, rot_y, 0f);

        Vector3 thisPos = transform.position;

        if (can_forward) // Observation 으로 나온 거로 penalty 처리 시도.
        {
            Vector2 forward = RotateVector2D(-rot_y, Vector2.up);
            if (teamType == TeamType.blue) // 청군은 느리다.
            {
                forward *= 0.5f;
            }

            Vector3 next_pos = new Vector3((float)Math.Round(thisPos.x + forward.x, 3), thisPos.y, (float)Math.Round(thisPos.z + forward.y, 3));

            if (act[0] == 1)
            {
                transform.position = next_pos;
            }
        }
        else
        {
            AddReward(penalty);
        }

        if (teamType == TeamType.red)
        {
            foreach (var goalTR in m_GameController.goalTRs)
            {
                if (Vector3.Distance(thisPos, goalTR.position) < 20f)
                {
                    m_GameController.TouchGoal(this);
                }
            }
        }
        else
        {
            float minDist = 100f;
            RiseAgent minRed = null;

            foreach (var red in m_GameController.reds)
            {
                if (red.gameObject.activeSelf == true)
                {
                    float tmpDist = Vector3.Distance(transform.position, red.transform.position);
                    if (tmpDist < minDist)
                    {
                        minDist = tmpDist;
                        minRed = red;
                    }
                }
            }

            if (minDist < 20f) // 2km 이내이고 가장 가까운애
            {
                AddReward(0.002f);
                m_GameController.MeetRedBlue(minRed);
            }
        }
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        AddReward(penalty);
        MoveAgent(actionBuffers.DiscreteActions);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;
        discreteActionsOut[1] = 0;
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 2;
        }
    }

    double[] GetCrossPoint(double[] a1, double[] a2, double[] b1, double[] b2)
    {
        if (Check(a1, a2, b1, b2))
        {
            double[] cross_point = new double[2];
            cross_point[0] = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0]) - (a1[0] - a2[0]) * (b1[0] * b2[1] - b1[1] * b2[0])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]));
            cross_point[1] = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] * b2[1] - b1[1] * b2[0])) / ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]));
            if (double.IsNaN(cross_point[0]) || double.IsNaN(cross_point[1]))
            {
                if (CompareBig(a1, a2))
                {
                    double[] tmp = a1;
                    a1 = a2;
                    a2 = tmp;
                }

                if (CompareBig(b1, b2))
                {
                    double[] tmp = b1;
                    b1 = b2;
                    b2 = tmp;
                }

                if (CompareSame(a2, b1))
                {
                    return a2;
                }
                else if (CompareSame(a1, b2))
                {
                    return a1;
                }
            }
            else
            {
                return cross_point;
            }
        }
        return null;
    }

    bool CompareBig(double[] a, double[] b)
    {
        if (a[0] > b[0])
        {
            return true;
        }
        else if (a[0] == b[0] && a[1] > b[1])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool CompareSame(double[] a, double[] b)
    {
        if (a[0] == b[0] && a[1] == b[1])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Check(double[] a1, double[] a2, double[] b1, double[] b2)
    {
        if (CCW(a1, a2, b1) * CCW(a1, a2, b2) == 0f)
        {
            if (CCW(b1, b2, a1) * CCW(b1, b2, a2) == 0f)
            {
                if (CompareBig(a1, a2))
                {
                    double[] tmp = a1;
                    a1 = a2;
                    a2 = tmp;
                }

                if (CompareBig(b1, b2))
                {
                    double[] tmp = b1;
                    b1 = b2;
                    b2 = tmp;
                }

                if ((CompareSame(a2, b1) || CompareBig(a2, b1)) && (CompareSame(b2, a1) || CompareBig(b2, a1)))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        if (CCW(a1, a2, b1) * CCW(a1, a2, b2) <= 0f)
        {
            if (CCW(b1, b2, a1) * CCW(b1, b2, a2) <= 0f)
            {
                return true;
            }
        }
        return false;
    }

    double CCW(double[] a, double[] b, double[] c)
    {
        return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    }
}
