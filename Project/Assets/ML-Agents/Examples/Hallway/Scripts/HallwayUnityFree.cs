using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using Random = UnityEngine.Random;

public enum TH_ray_type
{
    Nothing = 0,
    Goal_O = 1,
    Goal_X = 2,
    Wall = 3
}

public class HallwayUnityFree : Agent
{
    public Transform ground;
    public Transform area;
    public Transform symbolOGoal;
    public Transform symbolXGoal;
    public Transform symbolO;
    public Transform symbolX;

    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    HallwaySettings m_HallwaySettings;
    int rot_y;
    int m_Selection = 0;
    StatsRecorder m_statsRecorder;
    bool can_forward = true;
    const float kEpsilon = 0.00001F;

    public Vector2[][] line_points = {
        // 테두리
        new[] { new Vector2(-10f, 25f), new Vector2(10f, 25f) },
        new[] { new Vector2(10f, 25f), new Vector2(10f, -25f) },
        new[] { new Vector2(10f, -25f), new Vector2(-10f, -25f) },
        new[] { new Vector2(-10f, -25f), new Vector2(-10f, 25f) },
        // 장애물
        new[] { new Vector2(-2.5f, 0f), new Vector2(2.5f, 0f) }
    };

    public override void Initialize()
    {
        m_HallwaySettings = FindObjectOfType<HallwaySettings>();
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        m_statsRecorder = Academy.Instance.StatsRecorder;

        rot_y = (int)transform.eulerAngles.y;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_Selection); // Selection
        Vector2 my_pos_2d = new Vector2(transform.localPosition.x, transform.localPosition.z);
        can_forward = true;
        for (int i = -60; i <= 60; i += 10)
        {
            Vector2 rotatedVector = my_pos_2d + RotateVector2D(-rot_y + i, Vector2.up) * 100f;
            TH_ray_type ray_type = TH_ray_type.Nothing;
            float dist = 50f;

            Vector2 goal_O_pos_2d = new Vector2(symbolOGoal.position.x, symbolOGoal.position.z);
            Vector2[,] cross_goal_O_lines = { { goal_O_pos_2d + Vector2.up, goal_O_pos_2d + Vector2.down }, { goal_O_pos_2d + Vector2.right, goal_O_pos_2d + Vector2.left } };
            for (int j = 0; j < 2; ++j)
            {
                double[] _cross_point = GetCrossPoint(
                    new[] { (double)my_pos_2d.x, my_pos_2d.y },
                    new[] { (double)rotatedVector.x, rotatedVector.y },
                    new[] { (double)cross_goal_O_lines[j, 0].x, cross_goal_O_lines[j, 0].y },
                    new[] { (double)cross_goal_O_lines[j, 1].x, cross_goal_O_lines[j, 1].y }
                    );
                if (_cross_point != null)
                {
                    float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                    if (tmp_dist < dist)
                    {
                        dist = tmp_dist;
                        ray_type = TH_ray_type.Goal_O;
                    }
                }
            }

            Vector2 goal_X_pos_2d = new Vector2(symbolXGoal.position.x, symbolXGoal.position.z);
            Vector2[,] cross_goal_X_lines = { { goal_X_pos_2d + Vector2.up, goal_X_pos_2d + Vector2.down }, { goal_X_pos_2d + Vector2.right, goal_X_pos_2d + Vector2.left } };
            for (int j = 0; j < 2; ++j)
            {
                double[] _cross_point = GetCrossPoint(
                    new[] { (double)my_pos_2d.x, my_pos_2d.y },
                    new[] { (double)rotatedVector.x, rotatedVector.y },
                    new[] { (double)cross_goal_X_lines[j, 0].x, cross_goal_X_lines[j, 0].y },
                    new[] { (double)cross_goal_X_lines[j, 1].x, cross_goal_X_lines[j, 1].y }
                    );
                if (_cross_point != null)
                {
                    float tmp_dist = Vector2.Distance(my_pos_2d, new Vector2((float)_cross_point[0], (float)_cross_point[1]));
                    if (tmp_dist < dist)
                    {
                        dist = tmp_dist;
                        ray_type = TH_ray_type.Goal_X;
                    }
                }
            }

            foreach (Vector2[] linePoint in line_points)
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
                        if (tmp_dist < dist)
                        {
                            dist = tmp_dist;
                            ray_type = TH_ray_type.Wall;
                        }
                    }
                }
            }


            if (ray_type == TH_ray_type.Wall && dist < 2f) // 하나라도 정면에 걸리면 못감.
            {
                can_forward = false;
            }

            sensor.AddObservation(ray_type == TH_ray_type.Goal_O); // O
            sensor.AddObservation(ray_type == TH_ray_type.Goal_X); // X
            sensor.AddObservation(ray_type == TH_ray_type.Wall); // Wall
            sensor.AddObservation(dist / 50f);
        }
    }

    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        switch (act[1])
        {
            case 1:
                rot_y += 3;
                break;
            case 2:
                rot_y -= 3;
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

            Vector3 next_pos = new Vector3((float)Math.Round(thisPos.x + forward.x, 3), thisPos.y, (float)Math.Round(thisPos.z + forward.y, 3));

            if (act[0] == 1)
            {
                transform.position = next_pos;
            }
        }
        else
        {
            AddReward(-1f / MaxStep);
        }

        if (Vector3.Distance(thisPos, symbolOGoal.position) < 4f)
        {
            if (m_Selection == 0)
            {
                SetReward(1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.goalScoredMaterial, 0.5f));
                m_statsRecorder.Add("Goal/Correct", 1, StatAggregationMethod.Sum);
            }
            else
            {
                SetReward(-0.1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.failMaterial, 0.5f));
                m_statsRecorder.Add("Goal/Wrong", 1, StatAggregationMethod.Sum);
            }
            EndEpisode();
        }
        else if (Vector3.Distance(thisPos, symbolXGoal.position) < 4f)
        {
            if (m_Selection == 1)
            {
                SetReward(1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.goalScoredMaterial, 0.5f));
                m_statsRecorder.Add("Goal/Correct", 1, StatAggregationMethod.Sum);
            }
            else
            {
                SetReward(-0.1f);
                StartCoroutine(GoalScoredSwapGroundMaterial(m_HallwaySettings.failMaterial, 0.5f));
                m_statsRecorder.Add("Goal/Wrong", 1, StatAggregationMethod.Sum);
            }
            EndEpisode();
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        AddReward(-1f / MaxStep);
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

    public override void OnEpisodeBegin()
    {
        var agentOffset = -15f;
        m_Selection = Random.Range(0, 2);
        if (m_Selection == 0)
        {
            symbolO.position = new Vector3(0f, 2f, 0f) + ground.position;
            symbolX.position = new Vector3(0f, -1000f, 0f) + ground.position;
        }
        else
        {
            symbolO.position = new Vector3(0f, -1000f, 0f) + ground.position;
            symbolX.position = new Vector3(0f, 2f, 0f) + ground.position;
        }

        transform.position = new Vector3(0f + Random.Range(-3f, 3f),
            1f, agentOffset + Random.Range(-5f, 5f))
            + ground.position;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        var goalPos = Random.Range(0, 2);
        if (goalPos == 0)
        {
            symbolOGoal.position = new Vector3(7f, 0.5f, 22.29f) + area.position;
            symbolXGoal.position = new Vector3(-7f, 0.5f, 22.29f) + area.position;
        }
        else
        {
            symbolXGoal.position = new Vector3(7f, 0.5f, 22.29f) + area.position;
            symbolOGoal.position = new Vector3(-7f, 0.5f, 22.29f) + area.position;
        }
        m_statsRecorder.Add("Goal/Correct", 0, StatAggregationMethod.Sum);
        m_statsRecorder.Add("Goal/Wrong", 0, StatAggregationMethod.Sum);
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
}
