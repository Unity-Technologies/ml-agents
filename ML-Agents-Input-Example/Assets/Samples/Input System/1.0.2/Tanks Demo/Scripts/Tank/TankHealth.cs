using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;


public class TankHealth : MonoBehaviour
{
    [FormerlySerializedAs("m_StartingHealth")]
    public float startingHealth = 100f;                   // The amount of health each tank starts with.
    [FormerlySerializedAs("m_Slider")]
    public Slider slider;                                 // The slider to represent how much health the tank currently has.
    [FormerlySerializedAs("m_FillImage")]
    public Image fillImage;                               // The image component of the slider.
    [FormerlySerializedAs("m_FullHealthColor")]
    public Color fullHealthColor = Color.green;           // The color the health bar will be when on full health.
    [FormerlySerializedAs("m_ZeroHealthColor")]
    public Color zeroHealthColor = Color.red;             // The color the health bar will be when on no health.
    [FormerlySerializedAs("m_ExplosionPrefab")]
    public GameObject explosionPrefab;                    // A prefab that will be instantiated in Awake, then used whenever the tank dies.

    AudioSource m_ExplosionAudio;                   // The audio source to play when the tank explodes.
    ParticleSystem m_ExplosionParticles;            // The particle system the will play when the tank is destroyed.
    internal float currentHealth;                          // How much health the tank currently has.
    [HideInInspector]
    public bool isDead;                                    // Has the tank been reduced beyond zero health yet?

    void Awake()
    {
        // Instantiate the explosion prefab and get a reference to the particle system on it.
        m_ExplosionParticles = Instantiate(explosionPrefab).GetComponent<ParticleSystem>();

        // Get a reference to the audio source on the instantiated prefab.
        m_ExplosionAudio = m_ExplosionParticles.GetComponent<AudioSource>();

        // Disable the prefab so it can be activated when it's required.
        m_ExplosionParticles.gameObject.SetActive(false);
    }

    void OnEnable()
    {
        // When the tank is enabled, reset the tank's health and whether or not it's dead.
        currentHealth = startingHealth;
        isDead = false;

        // Update the health slider's value and color.
        SetHealthUI();
    }

    public void TakeDamage(float amount)
    {
        // Reduce current health by the amount of damage done.
        currentHealth -= amount;

        // Change the UI elements appropriately.
        SetHealthUI();

        // If the current health is at or below zero and it has not yet been registered, call OnDeath.
        if (currentHealth <= 0f && !isDead)
        {
            OnDeath();
        }
    }

    void SetHealthUI()
    {
        // Set the slider's value appropriately.
        slider.value = currentHealth;

        // Interpolate the color of the bar between the choosen colours based on the current percentage of the starting health.
        fillImage.color = Color.Lerp(zeroHealthColor, fullHealthColor, currentHealth / startingHealth);
    }

    void OnDeath()
    {
        // Set the flag so that this function is only called once.
        isDead = true;

        // Move the instantiated explosion prefab to the tank's position and turn it on.
        m_ExplosionParticles.transform.position = transform.position;
        m_ExplosionParticles.gameObject.SetActive(true);

        // Play the particle system of the tank exploding.
        m_ExplosionParticles.Play();

        // Play the tank explosion sound effect.
        // m_ExplosionAudio.Play();

        // Turn the tank off.
        gameObject.SetActive(false);
    }
}
