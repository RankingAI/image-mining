# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import ToxicImageDetection_pb2 as ToxicImageDetection__pb2


class NSFWStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.OpenNSFW = channel.unary_unary(
        '/NSFW/OpenNSFW',
        request_serializer=ToxicImageDetection__pb2.ImageURL.SerializeToString,
        response_deserializer=ToxicImageDetection__pb2.NSFWResult.FromString,
        )


class NSFWServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def OpenNSFW(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_NSFWServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'OpenNSFW': grpc.unary_unary_rpc_method_handler(
          servicer.OpenNSFW,
          request_deserializer=ToxicImageDetection__pb2.ImageURL.FromString,
          response_serializer=ToxicImageDetection__pb2.NSFWResult.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'NSFW', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))