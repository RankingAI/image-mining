# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ToxicImageDetection.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ToxicImageDetection.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x19ToxicImageDetection.proto\"\x18\n\x08ImageURL\x12\x0c\n\x04urls\x18\x01 \x03(\t\">\n\rProbabilities\x12\r\n\x05toxic\x18\x01 \x01(\x02\x12\x0e\n\x06sexual\x18\x02 \x01(\x02\x12\x0e\n\x06normal\x18\x03 \x01(\x02\",\n\nNSFWResult\x12\x1e\n\x06result\x18\x01 \x03(\x0b\x32\x0e.Probabilities2,\n\x04NSFW\x12$\n\x08OpenNSFW\x12\t.ImageURL\x1a\x0b.NSFWResult\"\x00\x62\x06proto3')
)




_IMAGEURL = _descriptor.Descriptor(
  name='ImageURL',
  full_name='ImageURL',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='urls', full_name='ImageURL.urls', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=29,
  serialized_end=53,
)


_PROBABILITIES = _descriptor.Descriptor(
  name='Probabilities',
  full_name='Probabilities',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='toxic', full_name='Probabilities.toxic', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sexual', full_name='Probabilities.sexual', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normal', full_name='Probabilities.normal', index=2,
      number=3, type=2, cpp_type=6, label=1,
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
  serialized_start=55,
  serialized_end=117,
)


_NSFWRESULT = _descriptor.Descriptor(
  name='NSFWResult',
  full_name='NSFWResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='NSFWResult.result', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=119,
  serialized_end=163,
)

_NSFWRESULT.fields_by_name['result'].message_type = _PROBABILITIES
DESCRIPTOR.message_types_by_name['ImageURL'] = _IMAGEURL
DESCRIPTOR.message_types_by_name['Probabilities'] = _PROBABILITIES
DESCRIPTOR.message_types_by_name['NSFWResult'] = _NSFWRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageURL = _reflection.GeneratedProtocolMessageType('ImageURL', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEURL,
  __module__ = 'ToxicImageDetection_pb2'
  # @@protoc_insertion_point(class_scope:ImageURL)
  ))
_sym_db.RegisterMessage(ImageURL)

Probabilities = _reflection.GeneratedProtocolMessageType('Probabilities', (_message.Message,), dict(
  DESCRIPTOR = _PROBABILITIES,
  __module__ = 'ToxicImageDetection_pb2'
  # @@protoc_insertion_point(class_scope:Probabilities)
  ))
_sym_db.RegisterMessage(Probabilities)

NSFWResult = _reflection.GeneratedProtocolMessageType('NSFWResult', (_message.Message,), dict(
  DESCRIPTOR = _NSFWRESULT,
  __module__ = 'ToxicImageDetection_pb2'
  # @@protoc_insertion_point(class_scope:NSFWResult)
  ))
_sym_db.RegisterMessage(NSFWResult)



_NSFW = _descriptor.ServiceDescriptor(
  name='NSFW',
  full_name='NSFW',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=165,
  serialized_end=209,
  methods=[
  _descriptor.MethodDescriptor(
    name='OpenNSFW',
    full_name='NSFW.OpenNSFW',
    index=0,
    containing_service=None,
    input_type=_IMAGEURL,
    output_type=_NSFWRESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_NSFW)

DESCRIPTOR.services_by_name['NSFW'] = _NSFW

# @@protoc_insertion_point(module_scope)
