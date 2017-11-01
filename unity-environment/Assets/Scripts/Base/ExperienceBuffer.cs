using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ExperienceBuffer
{
    public enum DataType
    {
        Float,
        Integer,
        Boolean
    }
    public struct DataInfo
    {
        public DataInfo(string name, DataType type, int unitLength)
        {
            this.type = type;
            this.unitLength = unitLength;
            this.name = name;
        }
        public string name;
        public DataType type;
        public int unitLength;
    }

    protected struct DataContainer
    {
        public DataContainer(DataInfo info, int maxLength)
        {
            this.info = info;
            if (info.type == DataType.Float)
            {
                dataList = new float[info.unitLength * maxLength];
            }
            else if (info.type == DataType.Integer)
            {
                dataList = new int[info.unitLength * maxLength];
            }
            else
            {
                dataList = new bool[info.unitLength * maxLength];
            }
        }
        public DataInfo info;
        public Array dataList;
    }


    protected Dictionary<string, DataContainer> dataset;

    private int maxCount = 0;
    private int nextBufferPointer = 0;
    private int replayBufferCount = 0;




    public ExperienceBuffer(int maxSize, params DataInfo[] dataInfos)
    {

        maxCount = maxSize;
        dataset = new Dictionary<string, DataContainer>();


        foreach (var i in dataInfos)
        {
            Debug.Assert(!dataset.ContainsKey(i.name));
            dataset[i.name] = new DataContainer(i, maxSize);
        }
    }




    /// <summary>
    /// Add data to the experience buffer
    /// </summary>
    /// <param name="data"></param>
    public void AddExperience(params Tuple<string, Array>[] data)
    {
        //check whether the input data are correct
        Debug.Assert(data.Length == dataset.Count, "Input data number is not the same as the experience buffer required");
        int size = data[0].Item2.Length / dataset[data[0].Item1].info.unitLength;
        foreach (var k in dataset.Keys)
        {
            bool found = false;
            foreach (var d in data)
            {
                if (d.Item1.Equals(k))
                {
                    found = true;
                    int newSize = size = d.Item2.Length / dataset[d.Item1].info.unitLength;
                    Debug.Assert(newSize == size, "The input Data has different sizes");
                }
            }
            Debug.Assert(found == true, "Data " + k + " is not fed to the experience buffer");
        }

        //feed the data.

        //add the episode to the buffer
        int numToAdd = size;
        int spaceLeft = maxCount - nextBufferPointer;

        int appendSize = Mathf.Min(spaceLeft, numToAdd);
        int fromStartSize = Mathf.Max(0, numToAdd - spaceLeft);

        foreach (var k in data)
        {
            DataContainer dd = dataset[k.Item1];
            //Debug.Log(k.Item1);
            //Debug.Log("add length " + k.Item2.Length + " copy length " + (appendSize * dd.info.unitLength).ToString());
            Array.Copy(k.Item2, 0, dd.dataList, nextBufferPointer * dd.info.unitLength, appendSize * dd.info.unitLength);
        }
        nextBufferPointer += appendSize;
        replayBufferCount += numToAdd;
        replayBufferCount = Mathf.Clamp(replayBufferCount, 0, maxCount);
        if (fromStartSize > 0)
        {
            foreach (var k in data)
            {
                DataContainer dd = dataset[k.Item1];
                Array.Copy(k.Item2, appendSize * dd.info.unitLength, dd.dataList, 0, fromStartSize * dd.info.unitLength);
            }
            nextBufferPointer = fromStartSize;
        }

    }

    public Type GetDataType(string key)
    {
        DataType type = dataset[key].info.type;
        if (type == DataType.Boolean)
        {
            return typeof(bool);
        }
        else if (type == DataType.Float)
        {
            return typeof(float);
        }
        else
        {
            return typeof(int);
        }
    }

    /// <summary>
    /// get samples form the buffer
    /// </summary>
    /// <param name="numOfSamples"></param>
    /// <param name="fetchAndOffset">tuple of <key of data to sample, sample index offset, returned dictionary key></param>
    /// <returns></returns>
    public Dictionary<string, Array> RandomSample(int numOfSamples, params Tuple<string, int, string>[] fetchAndOffset)
    {

        Dictionary<string, Array> result = new Dictionary<string, Array>();

        foreach (var d in fetchAndOffset)
        {
            Debug.Assert(dataset.ContainsKey(d.Item1));
            Debug.Assert(!result.ContainsKey(d.Item3));
            result[d.Item3] = Array.CreateInstance(GetDataType(d.Item1), dataset[d.Item1].info.unitLength * numOfSamples);
        }

        for (int i = 0; i < numOfSamples; ++i)
        {
            int sampleInd = UnityEngine.Random.Range(0, replayBufferCount);
            foreach (var d in fetchAndOffset)
            {
                DataContainer c = dataset[d.Item1];
                int unitLength = c.info.unitLength;
                int actSampleInd = (sampleInd + d.Item2) % replayBufferCount;
                Array.Copy(c.dataList, actSampleInd * unitLength, result[d.Item3], i * unitLength, unitLength);
            }
        }

        return result;
    }



    /// <summary>
    /// calculate the discrounted reward
    /// </summary>
    /// <param name="stepReward">the rewards list of each step</param>
    /// <param name="gamma">discount factor</param>
    /// <param name="nextValue">The reward after the last step of the list.</param>
    /// <returns></returns>
    public static float[] GetDiscountedRewards(List<float> stepReward, float gamma, float nextValue = 0)
    {
        float accum = nextValue;
        float[] result = new float[stepReward.Count];
        for (int i = stepReward.Count - 1; i >= 0; --i)
        {
            accum = accum * gamma + stepReward[i];
            result[i] = accum;
        }

        return result;
    }


    /// <summary>
    /// Get the Advantage for PPO algorithm
    /// </summary>
    /// <param name="stepRewards"></param>
    /// <param name="valueEstimates"></param>
    /// <param name="gamma"></param>
    /// <param name="lambda"></param>
    /// <param name="nextValue"></param>
    /// <returns></returns>
    public static float[] GetGAE(List<float> stepRewards, List<float> valueEstimates, float gamma, float lambda, float nextValue = 0)
    {
        Debug.Assert(stepRewards.Count == valueEstimates.Count, "stepReward and valueEstimates need to have the same length");

        int length = stepRewards.Count;
        List<float> deltaTs = new List<float>(length);
        for(int i = 0; i < length-1; ++i)
        {
            deltaTs.Add(stepRewards[i] + gamma * valueEstimates[i + 1] - valueEstimates[i]);
        }
        deltaTs.Add(stepRewards[length-1] + gamma * nextValue - valueEstimates[length-1]);

        float[] advantages = GetDiscountedRewards(deltaTs, gamma * lambda);
        return advantages;
    }

}