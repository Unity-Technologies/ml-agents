using System;
using System.Collections;
using NUnit.Framework;


namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class InplaceArrayTests
    {
        class LengthCases : IEnumerable
        {
            public IEnumerator GetEnumerator()
            {
                yield return 1;
                yield return 2;
                yield return 3;
                yield return 4;
            }
        }

        private InplaceArray<int> GetTestArray(int length)
        {
            switch (length)
            {
                case 1:
                    return new InplaceArray<int>(11);
                case 2:
                    return new InplaceArray<int>(11, 22);
                case 3:
                    return new InplaceArray<int>(11, 22, 33);
                case 4:
                    return new InplaceArray<int>(11, 22, 33, 44);
                default:
                    throw new ArgumentException("bad test!");
            }
        }

        private InplaceArray<int> GetZeroArray(int length)
        {
            switch (length)
            {
                case 1:
                    return new InplaceArray<int>(0);
                case 2:
                    return new InplaceArray<int>(0, 0);
                case 3:
                    return new InplaceArray<int>(0, 0, 0);
                case 4:
                    return new InplaceArray<int>(0, 0, 0, 0);
                default:
                    throw new ArgumentException("bad test!");
            }
        }

        [Test]
        public void TestInplaceArrayCtor()
        {
            var a1 = new InplaceArray<int>(11);
            Assert.AreEqual(1, a1.Length);
            Assert.AreEqual(11, a1[0]);

            var a2 = new InplaceArray<int>(11, 22);
            Assert.AreEqual(2, a2.Length);
            Assert.AreEqual(11, a2[0]);
            Assert.AreEqual(22, a2[1]);

            var a3 = new InplaceArray<int>(11, 22, 33);
            Assert.AreEqual(3, a3.Length);
            Assert.AreEqual(11, a3[0]);
            Assert.AreEqual(22, a3[1]);
            Assert.AreEqual(33, a3[2]);

            var a4 = new InplaceArray<int>(11, 22, 33, 44);
            Assert.AreEqual(4, a4.Length);
            Assert.AreEqual(11, a4[0]);
            Assert.AreEqual(22, a4[1]);
            Assert.AreEqual(33, a4[2]);
            Assert.AreEqual(44, a4[3]);
        }

        [TestCaseSource(typeof(LengthCases))]
        public void TestInplaceGetSet(int length)
        {
            var original = GetTestArray(length);

            for (var i = 0; i < original.Length; i++)
            {
                var modified = original;
                modified[i] = 0;
                for (var j = 0; j < original.Length; j++)
                {
                    if (i == j)
                    {
                        // This is the one we overwrote
                        Assert.AreEqual(0, modified[j]);
                    }
                    else
                    {
                        // Other elements should be unchanged
                        Assert.AreEqual(original[j], modified[j]);
                    }
                }
            }
        }

        [TestCaseSource(typeof(LengthCases))]
        public void TestInvalidAccess(int length)
        {
            var tmp = 0;
            var a = GetTestArray(length);
            // get
            Assert.Throws<IndexOutOfRangeException>(() => { tmp += a[-1]; });
            Assert.Throws<IndexOutOfRangeException>(() => { tmp += a[length]; });

            // set
            Assert.Throws<IndexOutOfRangeException>(() => { a[-1] = 0; });
            Assert.Throws<IndexOutOfRangeException>(() => { a[length] = 0; });

            // Make sure temp is used
            Assert.AreEqual(0, tmp);
        }

        [Test]
        public void TestOperatorEqualsDifferentLengths()
        {
            // Check arrays of different length are never equal (even if they have 0s in all elements)
            for (var l1 = 1; l1 <= 4; l1++)
            {
                var a1 = GetZeroArray(l1);
                for (var l2 = 1; l2 <= 4; l2++)
                {
                    var a2 = GetZeroArray(l2);
                    if (l1 == l2)
                    {
                        Assert.AreEqual(a1, a2);
                        Assert.IsTrue(a1 == a2);
                    }
                    else
                    {
                        Assert.AreNotEqual(a1, a2);
                        Assert.IsTrue(a1 != a2);
                    }
                }
            }
        }

        [TestCaseSource(typeof(LengthCases))]
        public void TestOperatorEquals(int length)
        {
            for (var index = 0; index < length; index++)
            {
                var a1 = GetTestArray(length);
                var a2 = GetTestArray(length);
                Assert.AreEqual(a1, a2);
                Assert.IsTrue(a1 == a2);

                a1[index] = 42;
                Assert.AreNotEqual(a1, a2);
                Assert.IsTrue(a1 != a2);

                a2[index] = 42;
                Assert.AreEqual(a1, a2);
                Assert.IsTrue(a1 == a2);
            }
        }

        [Test]
        public void TestToString()
        {
            Assert.AreEqual("[1]", new InplaceArray<int>(1).ToString());
            Assert.AreEqual("[1, 2]", new InplaceArray<int>(1, 2).ToString());
            Assert.AreEqual("[1, 2, 3]", new InplaceArray<int>(1, 2, 3).ToString());
            Assert.AreEqual("[1, 2, 3, 4]", new InplaceArray<int>(1, 2, 3, 4).ToString());
        }

        [TestCaseSource(typeof(LengthCases))]
        public void TestFromList(int length)
        {
            var intArray = new int[length];
            for (var i = 0; i < length; i++)
            {
                intArray[i] = (i + 1) * 11; // 11, 22, etc.
            }

            var converted = InplaceArray<int>.FromList(intArray);
            Assert.AreEqual(GetTestArray(length), converted);
        }
    }
}
