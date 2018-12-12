import sys
from grpc.tools import protoc

if __name__ == '__main__':

    if((len(sys.argv) != 3) | (sys.argv[1].endswith('proto') == False)):
        print('usage: {} proto_file_name code_output_dir'.format(sys.argv[0]))
        sys.exit(1)

    protoc.main(
        (
            '',
            '-I.',
            '--python_out={}'.format(sys.argv[2]),
            '--grpc_python_out={}'.format(sys.argv[2]),
            '{}'.format(sys.argv[1]),
        )
    )
