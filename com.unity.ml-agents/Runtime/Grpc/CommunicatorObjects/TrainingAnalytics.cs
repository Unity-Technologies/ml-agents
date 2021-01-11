// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: mlagents_envs/communicator_objects/training_analytics.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace Unity.MLAgents.CommunicatorObjects {

  /// <summary>Holder for reflection information generated from mlagents_envs/communicator_objects/training_analytics.proto</summary>
  internal static partial class TrainingAnalyticsReflection {

    #region Descriptor
    /// <summary>File descriptor for mlagents_envs/communicator_objects/training_analytics.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static TrainingAnalyticsReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "CjttbGFnZW50c19lbnZzL2NvbW11bmljYXRvcl9vYmplY3RzL3RyYWluaW5n",
            "X2FuYWx5dGljcy5wcm90bxIUY29tbXVuaWNhdG9yX29iamVjdHMi2AEKHlRy",
            "YWluaW5nRW52aXJvbm1lbnRJbml0aWFsaXplZBIYChBtbGFnZW50c192ZXJz",
            "aW9uGAEgASgJEh0KFW1sYWdlbnRzX2VudnNfdmVyc2lvbhgCIAEoCRIWCg5w",
            "eXRob25fdmVyc2lvbhgDIAEoCRIVCg10b3JjaF92ZXJzaW9uGAQgASgJEhkK",
            "EXRvcmNoX2RldmljZV90eXBlGAUgASgJEhAKCG51bV9lbnZzGAYgASgFEiEK",
            "GW51bV9yYW5kb21pemVkX3BhcmFtZXRlcnMYByABKAUi/QIKG1RyYWluaW5n",
            "QmVoYXZpb3JJbml0aWFsaXplZBIgChhleHRyaW5zaWNfcmV3YXJkX2VuYWJs",
            "ZWQYASABKAgSGwoTZ2FpbF9yZXdhcmRfZW5hYmxlZBgCIAEoCBIgChhjdXJp",
            "b3NpdHlfcmV3YXJkX2VuYWJsZWQYAyABKAgSGgoScm5kX3Jld2FyZF9lbmFi",
            "bGVkGAQgASgIEiIKGmJlaGF2aW9yYWxfY2xvbmluZ19lbmFibGVkGAUgASgI",
            "EhkKEXJlY3VycmVudF9lbmFibGVkGAYgASgIEhYKDnZpc3VhbF9lbmNvZGVy",
            "GAcgASgJEhoKEm51bV9uZXR3b3JrX2xheWVycxgIIAEoBRIgChhudW1fbmV0",
            "d29ya19oaWRkZW5fdW5pdHMYCSABKAUSGAoQdHJhaW5lcl90aHJlYWRlZBgK",
            "IAEoCBIZChFzZWxmX3BsYXlfZW5hYmxlZBgLIAEoCBIXCg91c2VzX2N1cnJp",
            "Y3VsdW0YDCABKAhCJaoCIlVuaXR5Lk1MQWdlbnRzLkNvbW11bmljYXRvck9i",
            "amVjdHNiBnByb3RvMw=="));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { },
          new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::Unity.MLAgents.CommunicatorObjects.TrainingEnvironmentInitialized), global::Unity.MLAgents.CommunicatorObjects.TrainingEnvironmentInitialized.Parser, new[]{ "MlagentsVersion", "MlagentsEnvsVersion", "PythonVersion", "TorchVersion", "TorchDeviceType", "NumEnvs", "NumRandomizedParameters" }, null, null, null),
            new pbr::GeneratedClrTypeInfo(typeof(global::Unity.MLAgents.CommunicatorObjects.TrainingBehaviorInitialized), global::Unity.MLAgents.CommunicatorObjects.TrainingBehaviorInitialized.Parser, new[]{ "ExtrinsicRewardEnabled", "GailRewardEnabled", "CuriosityRewardEnabled", "RndRewardEnabled", "BehavioralCloningEnabled", "RecurrentEnabled", "VisualEncoder", "NumNetworkLayers", "NumNetworkHiddenUnits", "TrainerThreaded", "SelfPlayEnabled", "UsesCurriculum" }, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  internal sealed partial class TrainingEnvironmentInitialized : pb::IMessage<TrainingEnvironmentInitialized> {
    private static readonly pb::MessageParser<TrainingEnvironmentInitialized> _parser = new pb::MessageParser<TrainingEnvironmentInitialized>(() => new TrainingEnvironmentInitialized());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<TrainingEnvironmentInitialized> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Unity.MLAgents.CommunicatorObjects.TrainingAnalyticsReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingEnvironmentInitialized() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingEnvironmentInitialized(TrainingEnvironmentInitialized other) : this() {
      mlagentsVersion_ = other.mlagentsVersion_;
      mlagentsEnvsVersion_ = other.mlagentsEnvsVersion_;
      pythonVersion_ = other.pythonVersion_;
      torchVersion_ = other.torchVersion_;
      torchDeviceType_ = other.torchDeviceType_;
      numEnvs_ = other.numEnvs_;
      numRandomizedParameters_ = other.numRandomizedParameters_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingEnvironmentInitialized Clone() {
      return new TrainingEnvironmentInitialized(this);
    }

    /// <summary>Field number for the "mlagents_version" field.</summary>
    public const int MlagentsVersionFieldNumber = 1;
    private string mlagentsVersion_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string MlagentsVersion {
      get { return mlagentsVersion_; }
      set {
        mlagentsVersion_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "mlagents_envs_version" field.</summary>
    public const int MlagentsEnvsVersionFieldNumber = 2;
    private string mlagentsEnvsVersion_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string MlagentsEnvsVersion {
      get { return mlagentsEnvsVersion_; }
      set {
        mlagentsEnvsVersion_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "python_version" field.</summary>
    public const int PythonVersionFieldNumber = 3;
    private string pythonVersion_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string PythonVersion {
      get { return pythonVersion_; }
      set {
        pythonVersion_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "torch_version" field.</summary>
    public const int TorchVersionFieldNumber = 4;
    private string torchVersion_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string TorchVersion {
      get { return torchVersion_; }
      set {
        torchVersion_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "torch_device_type" field.</summary>
    public const int TorchDeviceTypeFieldNumber = 5;
    private string torchDeviceType_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string TorchDeviceType {
      get { return torchDeviceType_; }
      set {
        torchDeviceType_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "num_envs" field.</summary>
    public const int NumEnvsFieldNumber = 6;
    private int numEnvs_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int NumEnvs {
      get { return numEnvs_; }
      set {
        numEnvs_ = value;
      }
    }

    /// <summary>Field number for the "num_randomized_parameters" field.</summary>
    public const int NumRandomizedParametersFieldNumber = 7;
    private int numRandomizedParameters_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int NumRandomizedParameters {
      get { return numRandomizedParameters_; }
      set {
        numRandomizedParameters_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as TrainingEnvironmentInitialized);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(TrainingEnvironmentInitialized other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (MlagentsVersion != other.MlagentsVersion) return false;
      if (MlagentsEnvsVersion != other.MlagentsEnvsVersion) return false;
      if (PythonVersion != other.PythonVersion) return false;
      if (TorchVersion != other.TorchVersion) return false;
      if (TorchDeviceType != other.TorchDeviceType) return false;
      if (NumEnvs != other.NumEnvs) return false;
      if (NumRandomizedParameters != other.NumRandomizedParameters) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (MlagentsVersion.Length != 0) hash ^= MlagentsVersion.GetHashCode();
      if (MlagentsEnvsVersion.Length != 0) hash ^= MlagentsEnvsVersion.GetHashCode();
      if (PythonVersion.Length != 0) hash ^= PythonVersion.GetHashCode();
      if (TorchVersion.Length != 0) hash ^= TorchVersion.GetHashCode();
      if (TorchDeviceType.Length != 0) hash ^= TorchDeviceType.GetHashCode();
      if (NumEnvs != 0) hash ^= NumEnvs.GetHashCode();
      if (NumRandomizedParameters != 0) hash ^= NumRandomizedParameters.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (MlagentsVersion.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(MlagentsVersion);
      }
      if (MlagentsEnvsVersion.Length != 0) {
        output.WriteRawTag(18);
        output.WriteString(MlagentsEnvsVersion);
      }
      if (PythonVersion.Length != 0) {
        output.WriteRawTag(26);
        output.WriteString(PythonVersion);
      }
      if (TorchVersion.Length != 0) {
        output.WriteRawTag(34);
        output.WriteString(TorchVersion);
      }
      if (TorchDeviceType.Length != 0) {
        output.WriteRawTag(42);
        output.WriteString(TorchDeviceType);
      }
      if (NumEnvs != 0) {
        output.WriteRawTag(48);
        output.WriteInt32(NumEnvs);
      }
      if (NumRandomizedParameters != 0) {
        output.WriteRawTag(56);
        output.WriteInt32(NumRandomizedParameters);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (MlagentsVersion.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(MlagentsVersion);
      }
      if (MlagentsEnvsVersion.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(MlagentsEnvsVersion);
      }
      if (PythonVersion.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(PythonVersion);
      }
      if (TorchVersion.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(TorchVersion);
      }
      if (TorchDeviceType.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(TorchDeviceType);
      }
      if (NumEnvs != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(NumEnvs);
      }
      if (NumRandomizedParameters != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(NumRandomizedParameters);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(TrainingEnvironmentInitialized other) {
      if (other == null) {
        return;
      }
      if (other.MlagentsVersion.Length != 0) {
        MlagentsVersion = other.MlagentsVersion;
      }
      if (other.MlagentsEnvsVersion.Length != 0) {
        MlagentsEnvsVersion = other.MlagentsEnvsVersion;
      }
      if (other.PythonVersion.Length != 0) {
        PythonVersion = other.PythonVersion;
      }
      if (other.TorchVersion.Length != 0) {
        TorchVersion = other.TorchVersion;
      }
      if (other.TorchDeviceType.Length != 0) {
        TorchDeviceType = other.TorchDeviceType;
      }
      if (other.NumEnvs != 0) {
        NumEnvs = other.NumEnvs;
      }
      if (other.NumRandomizedParameters != 0) {
        NumRandomizedParameters = other.NumRandomizedParameters;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            MlagentsVersion = input.ReadString();
            break;
          }
          case 18: {
            MlagentsEnvsVersion = input.ReadString();
            break;
          }
          case 26: {
            PythonVersion = input.ReadString();
            break;
          }
          case 34: {
            TorchVersion = input.ReadString();
            break;
          }
          case 42: {
            TorchDeviceType = input.ReadString();
            break;
          }
          case 48: {
            NumEnvs = input.ReadInt32();
            break;
          }
          case 56: {
            NumRandomizedParameters = input.ReadInt32();
            break;
          }
        }
      }
    }

  }

  internal sealed partial class TrainingBehaviorInitialized : pb::IMessage<TrainingBehaviorInitialized> {
    private static readonly pb::MessageParser<TrainingBehaviorInitialized> _parser = new pb::MessageParser<TrainingBehaviorInitialized>(() => new TrainingBehaviorInitialized());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<TrainingBehaviorInitialized> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Unity.MLAgents.CommunicatorObjects.TrainingAnalyticsReflection.Descriptor.MessageTypes[1]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingBehaviorInitialized() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingBehaviorInitialized(TrainingBehaviorInitialized other) : this() {
      extrinsicRewardEnabled_ = other.extrinsicRewardEnabled_;
      gailRewardEnabled_ = other.gailRewardEnabled_;
      curiosityRewardEnabled_ = other.curiosityRewardEnabled_;
      rndRewardEnabled_ = other.rndRewardEnabled_;
      behavioralCloningEnabled_ = other.behavioralCloningEnabled_;
      recurrentEnabled_ = other.recurrentEnabled_;
      visualEncoder_ = other.visualEncoder_;
      numNetworkLayers_ = other.numNetworkLayers_;
      numNetworkHiddenUnits_ = other.numNetworkHiddenUnits_;
      trainerThreaded_ = other.trainerThreaded_;
      selfPlayEnabled_ = other.selfPlayEnabled_;
      usesCurriculum_ = other.usesCurriculum_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public TrainingBehaviorInitialized Clone() {
      return new TrainingBehaviorInitialized(this);
    }

    /// <summary>Field number for the "extrinsic_reward_enabled" field.</summary>
    public const int ExtrinsicRewardEnabledFieldNumber = 1;
    private bool extrinsicRewardEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool ExtrinsicRewardEnabled {
      get { return extrinsicRewardEnabled_; }
      set {
        extrinsicRewardEnabled_ = value;
      }
    }

    /// <summary>Field number for the "gail_reward_enabled" field.</summary>
    public const int GailRewardEnabledFieldNumber = 2;
    private bool gailRewardEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool GailRewardEnabled {
      get { return gailRewardEnabled_; }
      set {
        gailRewardEnabled_ = value;
      }
    }

    /// <summary>Field number for the "curiosity_reward_enabled" field.</summary>
    public const int CuriosityRewardEnabledFieldNumber = 3;
    private bool curiosityRewardEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool CuriosityRewardEnabled {
      get { return curiosityRewardEnabled_; }
      set {
        curiosityRewardEnabled_ = value;
      }
    }

    /// <summary>Field number for the "rnd_reward_enabled" field.</summary>
    public const int RndRewardEnabledFieldNumber = 4;
    private bool rndRewardEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool RndRewardEnabled {
      get { return rndRewardEnabled_; }
      set {
        rndRewardEnabled_ = value;
      }
    }

    /// <summary>Field number for the "behavioral_cloning_enabled" field.</summary>
    public const int BehavioralCloningEnabledFieldNumber = 5;
    private bool behavioralCloningEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool BehavioralCloningEnabled {
      get { return behavioralCloningEnabled_; }
      set {
        behavioralCloningEnabled_ = value;
      }
    }

    /// <summary>Field number for the "recurrent_enabled" field.</summary>
    public const int RecurrentEnabledFieldNumber = 6;
    private bool recurrentEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool RecurrentEnabled {
      get { return recurrentEnabled_; }
      set {
        recurrentEnabled_ = value;
      }
    }

    /// <summary>Field number for the "visual_encoder" field.</summary>
    public const int VisualEncoderFieldNumber = 7;
    private string visualEncoder_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string VisualEncoder {
      get { return visualEncoder_; }
      set {
        visualEncoder_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "num_network_layers" field.</summary>
    public const int NumNetworkLayersFieldNumber = 8;
    private int numNetworkLayers_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int NumNetworkLayers {
      get { return numNetworkLayers_; }
      set {
        numNetworkLayers_ = value;
      }
    }

    /// <summary>Field number for the "num_network_hidden_units" field.</summary>
    public const int NumNetworkHiddenUnitsFieldNumber = 9;
    private int numNetworkHiddenUnits_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int NumNetworkHiddenUnits {
      get { return numNetworkHiddenUnits_; }
      set {
        numNetworkHiddenUnits_ = value;
      }
    }

    /// <summary>Field number for the "trainer_threaded" field.</summary>
    public const int TrainerThreadedFieldNumber = 10;
    private bool trainerThreaded_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool TrainerThreaded {
      get { return trainerThreaded_; }
      set {
        trainerThreaded_ = value;
      }
    }

    /// <summary>Field number for the "self_play_enabled" field.</summary>
    public const int SelfPlayEnabledFieldNumber = 11;
    private bool selfPlayEnabled_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool SelfPlayEnabled {
      get { return selfPlayEnabled_; }
      set {
        selfPlayEnabled_ = value;
      }
    }

    /// <summary>Field number for the "uses_curriculum" field.</summary>
    public const int UsesCurriculumFieldNumber = 12;
    private bool usesCurriculum_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool UsesCurriculum {
      get { return usesCurriculum_; }
      set {
        usesCurriculum_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as TrainingBehaviorInitialized);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(TrainingBehaviorInitialized other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (ExtrinsicRewardEnabled != other.ExtrinsicRewardEnabled) return false;
      if (GailRewardEnabled != other.GailRewardEnabled) return false;
      if (CuriosityRewardEnabled != other.CuriosityRewardEnabled) return false;
      if (RndRewardEnabled != other.RndRewardEnabled) return false;
      if (BehavioralCloningEnabled != other.BehavioralCloningEnabled) return false;
      if (RecurrentEnabled != other.RecurrentEnabled) return false;
      if (VisualEncoder != other.VisualEncoder) return false;
      if (NumNetworkLayers != other.NumNetworkLayers) return false;
      if (NumNetworkHiddenUnits != other.NumNetworkHiddenUnits) return false;
      if (TrainerThreaded != other.TrainerThreaded) return false;
      if (SelfPlayEnabled != other.SelfPlayEnabled) return false;
      if (UsesCurriculum != other.UsesCurriculum) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (ExtrinsicRewardEnabled != false) hash ^= ExtrinsicRewardEnabled.GetHashCode();
      if (GailRewardEnabled != false) hash ^= GailRewardEnabled.GetHashCode();
      if (CuriosityRewardEnabled != false) hash ^= CuriosityRewardEnabled.GetHashCode();
      if (RndRewardEnabled != false) hash ^= RndRewardEnabled.GetHashCode();
      if (BehavioralCloningEnabled != false) hash ^= BehavioralCloningEnabled.GetHashCode();
      if (RecurrentEnabled != false) hash ^= RecurrentEnabled.GetHashCode();
      if (VisualEncoder.Length != 0) hash ^= VisualEncoder.GetHashCode();
      if (NumNetworkLayers != 0) hash ^= NumNetworkLayers.GetHashCode();
      if (NumNetworkHiddenUnits != 0) hash ^= NumNetworkHiddenUnits.GetHashCode();
      if (TrainerThreaded != false) hash ^= TrainerThreaded.GetHashCode();
      if (SelfPlayEnabled != false) hash ^= SelfPlayEnabled.GetHashCode();
      if (UsesCurriculum != false) hash ^= UsesCurriculum.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (ExtrinsicRewardEnabled != false) {
        output.WriteRawTag(8);
        output.WriteBool(ExtrinsicRewardEnabled);
      }
      if (GailRewardEnabled != false) {
        output.WriteRawTag(16);
        output.WriteBool(GailRewardEnabled);
      }
      if (CuriosityRewardEnabled != false) {
        output.WriteRawTag(24);
        output.WriteBool(CuriosityRewardEnabled);
      }
      if (RndRewardEnabled != false) {
        output.WriteRawTag(32);
        output.WriteBool(RndRewardEnabled);
      }
      if (BehavioralCloningEnabled != false) {
        output.WriteRawTag(40);
        output.WriteBool(BehavioralCloningEnabled);
      }
      if (RecurrentEnabled != false) {
        output.WriteRawTag(48);
        output.WriteBool(RecurrentEnabled);
      }
      if (VisualEncoder.Length != 0) {
        output.WriteRawTag(58);
        output.WriteString(VisualEncoder);
      }
      if (NumNetworkLayers != 0) {
        output.WriteRawTag(64);
        output.WriteInt32(NumNetworkLayers);
      }
      if (NumNetworkHiddenUnits != 0) {
        output.WriteRawTag(72);
        output.WriteInt32(NumNetworkHiddenUnits);
      }
      if (TrainerThreaded != false) {
        output.WriteRawTag(80);
        output.WriteBool(TrainerThreaded);
      }
      if (SelfPlayEnabled != false) {
        output.WriteRawTag(88);
        output.WriteBool(SelfPlayEnabled);
      }
      if (UsesCurriculum != false) {
        output.WriteRawTag(96);
        output.WriteBool(UsesCurriculum);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (ExtrinsicRewardEnabled != false) {
        size += 1 + 1;
      }
      if (GailRewardEnabled != false) {
        size += 1 + 1;
      }
      if (CuriosityRewardEnabled != false) {
        size += 1 + 1;
      }
      if (RndRewardEnabled != false) {
        size += 1 + 1;
      }
      if (BehavioralCloningEnabled != false) {
        size += 1 + 1;
      }
      if (RecurrentEnabled != false) {
        size += 1 + 1;
      }
      if (VisualEncoder.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(VisualEncoder);
      }
      if (NumNetworkLayers != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(NumNetworkLayers);
      }
      if (NumNetworkHiddenUnits != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(NumNetworkHiddenUnits);
      }
      if (TrainerThreaded != false) {
        size += 1 + 1;
      }
      if (SelfPlayEnabled != false) {
        size += 1 + 1;
      }
      if (UsesCurriculum != false) {
        size += 1 + 1;
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(TrainingBehaviorInitialized other) {
      if (other == null) {
        return;
      }
      if (other.ExtrinsicRewardEnabled != false) {
        ExtrinsicRewardEnabled = other.ExtrinsicRewardEnabled;
      }
      if (other.GailRewardEnabled != false) {
        GailRewardEnabled = other.GailRewardEnabled;
      }
      if (other.CuriosityRewardEnabled != false) {
        CuriosityRewardEnabled = other.CuriosityRewardEnabled;
      }
      if (other.RndRewardEnabled != false) {
        RndRewardEnabled = other.RndRewardEnabled;
      }
      if (other.BehavioralCloningEnabled != false) {
        BehavioralCloningEnabled = other.BehavioralCloningEnabled;
      }
      if (other.RecurrentEnabled != false) {
        RecurrentEnabled = other.RecurrentEnabled;
      }
      if (other.VisualEncoder.Length != 0) {
        VisualEncoder = other.VisualEncoder;
      }
      if (other.NumNetworkLayers != 0) {
        NumNetworkLayers = other.NumNetworkLayers;
      }
      if (other.NumNetworkHiddenUnits != 0) {
        NumNetworkHiddenUnits = other.NumNetworkHiddenUnits;
      }
      if (other.TrainerThreaded != false) {
        TrainerThreaded = other.TrainerThreaded;
      }
      if (other.SelfPlayEnabled != false) {
        SelfPlayEnabled = other.SelfPlayEnabled;
      }
      if (other.UsesCurriculum != false) {
        UsesCurriculum = other.UsesCurriculum;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 8: {
            ExtrinsicRewardEnabled = input.ReadBool();
            break;
          }
          case 16: {
            GailRewardEnabled = input.ReadBool();
            break;
          }
          case 24: {
            CuriosityRewardEnabled = input.ReadBool();
            break;
          }
          case 32: {
            RndRewardEnabled = input.ReadBool();
            break;
          }
          case 40: {
            BehavioralCloningEnabled = input.ReadBool();
            break;
          }
          case 48: {
            RecurrentEnabled = input.ReadBool();
            break;
          }
          case 58: {
            VisualEncoder = input.ReadString();
            break;
          }
          case 64: {
            NumNetworkLayers = input.ReadInt32();
            break;
          }
          case 72: {
            NumNetworkHiddenUnits = input.ReadInt32();
            break;
          }
          case 80: {
            TrainerThreaded = input.ReadBool();
            break;
          }
          case 88: {
            SelfPlayEnabled = input.ReadBool();
            break;
          }
          case 96: {
            UsesCurriculum = input.ReadBool();
            break;
          }
        }
      }
    }

  }

  #endregion

}

#endregion Designer generated code
