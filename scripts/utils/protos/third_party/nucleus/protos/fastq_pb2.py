# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: third_party/nucleus/protos/fastq.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='third_party/nucleus/protos/fastq.proto',
  package='nucleus.genomics.v1',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n&third_party/nucleus/protos/fastq.proto\x12\x13nucleus.genomics.v1\"Q\n\x0b\x46\x61stqRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x10\n\x08sequence\x18\x03 \x01(\t\x12\x0f\n\x07quality\x18\x04 \x01(\t\"8\n\x12\x46\x61stqReaderOptions\x12\x1c\n\x14skip_invalid_records\x18\x02 \x01(\x08J\x04\x08\x01\x10\x02\"\x14\n\x12\x46\x61stqWriterOptionsb\x06proto3')
)




_FASTQRECORD = _descriptor.Descriptor(
  name='FastqRecord',
  full_name='nucleus.genomics.v1.FastqRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='nucleus.genomics.v1.FastqRecord.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='nucleus.genomics.v1.FastqRecord.description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sequence', full_name='nucleus.genomics.v1.FastqRecord.sequence', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='quality', full_name='nucleus.genomics.v1.FastqRecord.quality', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=63,
  serialized_end=144,
)


_FASTQREADEROPTIONS = _descriptor.Descriptor(
  name='FastqReaderOptions',
  full_name='nucleus.genomics.v1.FastqReaderOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='skip_invalid_records', full_name='nucleus.genomics.v1.FastqReaderOptions.skip_invalid_records', index=0,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=146,
  serialized_end=202,
)


_FASTQWRITEROPTIONS = _descriptor.Descriptor(
  name='FastqWriterOptions',
  full_name='nucleus.genomics.v1.FastqWriterOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
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
  serialized_start=204,
  serialized_end=224,
)

DESCRIPTOR.message_types_by_name['FastqRecord'] = _FASTQRECORD
DESCRIPTOR.message_types_by_name['FastqReaderOptions'] = _FASTQREADEROPTIONS
DESCRIPTOR.message_types_by_name['FastqWriterOptions'] = _FASTQWRITEROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FastqRecord = _reflection.GeneratedProtocolMessageType('FastqRecord', (_message.Message,), {
  'DESCRIPTOR' : _FASTQRECORD,
  '__module__' : 'third_party.nucleus.protos.fastq_pb2'
  # @@protoc_insertion_point(class_scope:nucleus.genomics.v1.FastqRecord)
  })
_sym_db.RegisterMessage(FastqRecord)

FastqReaderOptions = _reflection.GeneratedProtocolMessageType('FastqReaderOptions', (_message.Message,), {
  'DESCRIPTOR' : _FASTQREADEROPTIONS,
  '__module__' : 'third_party.nucleus.protos.fastq_pb2'
  # @@protoc_insertion_point(class_scope:nucleus.genomics.v1.FastqReaderOptions)
  })
_sym_db.RegisterMessage(FastqReaderOptions)

FastqWriterOptions = _reflection.GeneratedProtocolMessageType('FastqWriterOptions', (_message.Message,), {
  'DESCRIPTOR' : _FASTQWRITEROPTIONS,
  '__module__' : 'third_party.nucleus.protos.fastq_pb2'
  # @@protoc_insertion_point(class_scope:nucleus.genomics.v1.FastqWriterOptions)
  })
_sym_db.RegisterMessage(FastqWriterOptions)


# @@protoc_insertion_point(module_scope)
