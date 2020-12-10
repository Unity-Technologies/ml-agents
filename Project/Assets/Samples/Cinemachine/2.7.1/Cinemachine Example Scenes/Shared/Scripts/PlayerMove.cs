using System;
using Cinemachine.Utility;
using UnityEngine;

public class PlayerMove : MonoBehaviour
{
    public float Speed;
    public float VelocityDamping;
    public float JumpTime;

    public enum ForwardMode { Camera, Player, World };
    public ForwardMode InputForward;

    public bool RotatePlayer = true;

    public Action SpaceAction;
    public Action EnterAction;

    Vector3 m_currentVleocity;
    float m_currentJumpSpeed;
    float m_restY;

    private void Reset()
    {
        Speed = 5;
        InputForward = ForwardMode.Camera;
        RotatePlayer = true;
        VelocityDamping = 0.5f;
        m_currentVleocity = Vector3.zero;
        JumpTime = 1;
        m_currentJumpSpeed = 0;
    }

    private void OnEnable()
    {
        m_currentJumpSpeed = 0;
        m_restY = transform.position.y;
        SpaceAction -= Jump;
        SpaceAction += Jump;
    }

    void Update()
    {
        Vector3 fwd;
        switch (InputForward)
        {
            case ForwardMode.Camera: fwd = Camera.main.transform.forward; break;
            case ForwardMode.Player: fwd = transform.forward; break;
            case ForwardMode.World: default: fwd = Vector3.forward; break;
        }

        fwd.y = 0;
        fwd = fwd.normalized;
        if (fwd.sqrMagnitude < 0.01f)
            return;

        Quaternion inputFrame = Quaternion.LookRotation(fwd, Vector3.up);
        Vector3 input = new Vector3(Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical"));
        input = inputFrame * input;

        var dt = Time.deltaTime;
        var desiredVelocity = input * Speed;
        var deltaVel = desiredVelocity - m_currentVleocity;
        m_currentVleocity += Damper.Damp(deltaVel, VelocityDamping, dt);

        transform.position += m_currentVleocity * dt;
        if (RotatePlayer && m_currentVleocity.sqrMagnitude > 0.01f)
        {
            var qA = transform.rotation;
            var qB = Quaternion.LookRotation(
                (InputForward == ForwardMode.Player && Vector3.Dot(fwd, m_currentVleocity) < 0)
                    ? -m_currentVleocity : m_currentVleocity);
            transform.rotation = Quaternion.Slerp(qA, qB, Damper.Damp(1, VelocityDamping, dt));
        }

        // Process jump
        if (m_currentJumpSpeed != 0)
            m_currentJumpSpeed -= 10 * dt;
        var p = transform.position;
        p.y += m_currentJumpSpeed * dt;
        if (p.y < m_restY)
        {
            p.y = m_restY;
            m_currentJumpSpeed = 0;
        }
        transform.position = p;

        if (Input.GetKeyDown(KeyCode.Space) && SpaceAction != null)
            SpaceAction();
        if (Input.GetKeyDown(KeyCode.Return) && EnterAction != null)
            EnterAction();
    }

    public void Jump() { m_currentJumpSpeed += 10 * JumpTime * 0.5f; }
}
