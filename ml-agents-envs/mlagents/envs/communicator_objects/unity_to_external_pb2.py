# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlagents/envs/communicator_objects/unity_to_external.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlagents.envs.communicator_objects import unity_message_pb2 as mlagents_dot_envs_dot_communicator__objects_dot_unity__message__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlagents/envs/communicator_objects/unity_to_external.proto',
  package='communicator_objects',
  syntax='proto3',
  serialized_pb=_b('\n:mlagents/envs/communicator_objects/unity_to_external.proto\x12\x14\x63ommunicator_objects\x1a\x36mlagents/envs/communicator_objects/unity_message.proto2g\n\x0fUnityToExternal\x12T\n\x08\x45xchange\x12\".communicator_objects.UnityMessage\x1a\".communicator_objects.UnityMessage\"\x00\x42\x1f\xaa\x02\x1cMLAgents.CommunicatorObjectsb\x06proto3')
  ,
  dependencies=[mlagents_dot_envs_dot_communicator__objects_dot_unity__message__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\252\002\034MLAgents.CommunicatorObjects'))

_UNITYTOEXTERNAL = _descriptor.ServiceDescriptor(
  name='UnityToExternal',
  full_name='communicator_objects.UnityToExternal',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=140,
  serialized_end=243,
  methods=[
  _descriptor.MethodDescriptor(
    name='Exchange',
    full_name='communicator_objects.UnityToExternal.Exchange',
    index=0,
    containing_service=None,
    input_type=mlagents_dot_envs_dot_communicator__objects_dot_unity__message__pb2._UNITYMESSAGE,
    output_type=mlagents_dot_envs_dot_communicator__objects_dot_unity__message__pb2._UNITYMESSAGE,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_UNITYTOEXTERNAL)

DESCRIPTOR.services_by_name['UnityToExternal'] = _UNITYTOEXTERNAL

# @@protoc_insertion_point(module_scope)
