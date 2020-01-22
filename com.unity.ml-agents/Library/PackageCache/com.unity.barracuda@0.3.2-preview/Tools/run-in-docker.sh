function build_docker_image()
{
    echo "Updating docker image..."
    docker build -t $NAME "$DIR" -f "$DIR/Dockerfile" > dockerbuild.log
}

function clean_docker_container()
{
    echo "Cleaning docker container..."
    docker stop $NAME > /dev/null
    docker rm $NAME > /dev/null
}

function start_docker_container()
{
    echo "Starting docker container..."
    docker run --detach --name $NAME -t $NAME
}

function docker_cp()
{
    docker cp "$1" "$2"
}

function docker_exec_python3()
{
    docker exec -it $NAME python3 "$1" "$2" "$3" $4
}

HELP_INVOCATION=0

if [ $# -eq 2 ] && { [ "$2" = "-h" ] || [ "$2" = "--help" ]; }
then
    HELP_INVOCATION=1
fi

if [ $# -le  2 ] && [ $HELP_INVOCATION -eq 0 ]
then
    echo "Usage: run-in-docker.sh <local_script> <src_file> <dst_file> [other_flags]"
    exit -1;
fi

#Remaining args
ARGS=""
for ((i = 4; i <= $#; i++ )); do
    ARGS="$ARGS \"${!i}\""
done

SCRIPT_NAME=$1

if [ $HELP_INVOCATION -eq 0 ]
then
    SRC_FILE=$2
    DST_FILE=$3
    SCRIPT_FILE_NAME=`basename "$SCRIPT_NAME"`
    SRC_FILE_NAME=`basename "$SRC_FILE"`
    DST_FILE_NAME=`basename "$DST_FILE"`

    if [ "$SRC_FILE_NAME" = "$DST_FILE_NAME" ]
    then
        echo "Source \"$SRC_FILE_NAME\" and destination \"$DST_FILE_NAME\" file names should NOT match!"
        exit -1
    fi
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DST=/Barracuda
DST_TOOLS=$DST/Tools
NAME=barconv

#echo "SCRIPT_NAME=$SCRIPT_NAME"
#echo "SRC_FILE=$SRC_FILE"
#echo "DST_FILE=$DST_FILE"
#echo "SRC_FILE_NAME=$SRC_FILE_NAME"
#echo "DST_FILE_NAME=$DST_FILE_NAME"
#echo "ARGS=$ARGS"


build_docker_image

clean_docker_container
start_docker_container

if [ $HELP_INVOCATION -eq 1 ]
then
    docker_exec_python3 $SCRIPT_NAME --help
    exit 0
fi

echo "Copying files to docker container..."
docker_cp "$DIR/" "$NAME:$DST/"
docker_cp "$SCRIPT_NAME" "$NAME:$DST_TOOLS/"
docker_cp "$SRC_FILE" "$NAME:$DST_TOOLS/"

echo "Converting in docker..."
docker_exec_python3 "$DST_TOOLS/$SCRIPT_FILE_NAME" "$DST_TOOLS/$SRC_FILE_NAME" "$DST_TOOLS/$DST_FILE_NAME" "$ARGS"

echo "Getting files from docker..."
docker_cp "$NAME:$DST_TOOLS/$DST_FILE_NAME" "$DST_FILE"