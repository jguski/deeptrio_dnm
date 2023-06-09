# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: third_party/nucleus/protos/bedgraph.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='third_party/nucleus/protos/bedgraph.proto',
  package='nucleus.genomics.v1',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n)third_party/nucleus/protos/bedgraph.proto\x12\x13nucleus.genomics.v1\"X\n\x0e\x42\x65\x64GraphRecord\x12\x16\n\x0ereference_name\x18\x01 \x01(\t\x12\r\n\x05start\x18\x02 \x01(\x03\x12\x0b\n\x03\x65nd\x18\x03 \x01(\x03\x12\x12\n\ndata_value\x18\x04 \x01(\x01\x62\x06proto3')
)




_BEDGRAPHRECORD = _descriptor.Descriptor(
  name='BedGraphRecord',
  full_name='nucleus.genomics.v1.BedGraphRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reference_name', full_name='nucleus.genomics.v1.BedGraphRecord.reference_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='nucleus.genomics.v1.BedGraphRecord.start', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='nucleus.genomics.v1.BedGraphRecord.end', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_value', full_name='nucleus.genomics.v1.BedGraphRecord.data_value', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=66,
  serialized_end=154,
)

DESCRIPTOR.message_types_by_name['BedGraphRecord'] = _BEDGRAPHRECORD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BedGraphRecord = _reflection.GeneratedProtocolMessageType('BedGraphRecord', (_message.Message,), {
  'DESCRIPTOR' : _BEDGRAPHRECORD,
  '__module__' : 'third_party.nucleus.protos.bedgraph_pb2'
  # @@protoc_insertion_point(class_scope:nucleus.genomics.v1.BedGraphRecord)
  })
_sym_db.RegisterMessage(BedGraphRecord)


# @@protoc_insertion_point(module_scope)
