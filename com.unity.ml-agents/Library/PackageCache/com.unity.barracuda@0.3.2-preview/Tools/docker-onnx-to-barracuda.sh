DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

$DIR/run-in-docker.sh "$DIR/onnx_to_barracuda.py" $@
