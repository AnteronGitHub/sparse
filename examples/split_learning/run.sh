export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

docker container stop training-server training-local
docker network rm sparse

docker network create sparse

docker run --rm --network sparse -e WORKER_LISTEN_ADDRESS=0.0.0.0 --name training-server split_learning:server.amd64 &
docker run --rm --network sparse -e MASTER_UPSTREAM_HOST=training-server --name training-local -it split_learning:client.amd64
