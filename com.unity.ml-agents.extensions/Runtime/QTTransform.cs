using UnityEngine;

namespace Unity.MLAgents.Extensions
{
    /// <summary>
    /// Simple linear transform representation consisting of a rotation and translation.
    /// When deriving the math, it can be helpful to think of the transform as a 4x4 block matrix:
    /// <example>
    /// | R | t |
    /// ----+----
    /// | 0 | 1 |
    /// </example>
    /// where R is a 3x3 rotation, t is a 3x1 translation, 0 is a 1x3 vector of 0s.
    /// </summary>
    public struct QTTransform
    {
        public Quaternion Rotation;
        public Vector3 Translation;

        /// <summary>
        /// Multiply two transforms.
        /// <example>
        /// | R1 | t1 |    | R2 | t2 |   | R1*R2 | R1*t2 + t1 |
        /// -----+----- *  -----+----- = --------+-------------
        /// | 0  | 1  |    | 0  | 1  |   |   0   |      1     |
        /// </example>
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <returns></returns>
        public static QTTransform operator *(QTTransform t1, QTTransform t2)
        {
            var translation = (t1.Rotation * t2.Translation) + t1.Translation;
            var rotation = t1.Rotation * t2.Rotation;
            return new QTTransform { Rotation = rotation, Translation = translation };
        }

        public QTTransform Inverse()
        {
            var rotationInverse = Quaternion.Inverse(Rotation);
            var translationInverse = -(rotationInverse * Translation);
            return new QTTransform { Rotation = rotationInverse, Translation = translationInverse };
        }

        public static QTTransform Identity
        {
            get { return new QTTransform { Rotation = Quaternion.identity, Translation = Vector3.zero}; }
        }

        // TODO optimize inv(A)*B?

    }
}
