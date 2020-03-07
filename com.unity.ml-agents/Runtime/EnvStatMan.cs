using System;
using UnityEngine;
using System.Collections.Generic;
using Google.Protobuf.Collections;

namespace MLAgents
{
    public class EnvStatMan
    {
        Dictionary<string, float> floatStat;
        Dictionary<string, string> stringStat;
        public EnvStatMan()
        {
            Reset();
        }
        public void Reset()
        {
            floatStat = new Dictionary<string, float>();
            stringStat = new Dictionary<string, string>();
        }
        static string tbprefix = "tb:";
        public void AddFloatStat(string key,float valf)
        {
            floatStat[tbprefix+key] = valf;
        }
        public void AddStringStat(string key, string vals)
        {
            stringStat[key] = vals;
        }
        public void FillFloatMapField(MapField<string,float> mapfield)
        {
            foreach( var k in floatStat.Keys)
            {
                mapfield[k] = floatStat[k];
            }
        }
        public void FillStringMapField(MapField<string, string> mapfield)
        {
            foreach (var k in stringStat.Keys)
            {
                mapfield[k] = stringStat[k];
            }
        }
    }
}