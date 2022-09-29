uid := $(shell id -u)
base_image_file := $(abspath ../../.DOCKER)
py_sources := $(shell find . -iname *.py)

docker_image := split_learning
docker_build_file := .DOCKER

docker_run := docker run \
               --network host \
               -v /run/jtop.sock:/run/jtop.sock \
		       -v $(abspath ../../src):/usr/lib/sparse \
		       -v $(abspath ../../data):/data \
		       -v $(abspath .):/app

.PHONY: all
all: $(docker_build_file)

.PHONY: run-aio
run-aio: | $(docker_build_file)
	$(docker_run) \
		--name learning_aio \
		--rm -it $(docker_image)

.PHONY: run-data-source
run-data-source: | $(docker_build_file)
	$(docker_run) \
		--name learning_data_source \
		--rm -it $(docker_image) \
		python3 benchmark.py --suite offload_source --batches 64 --batch-size 64

.PHONY: run-unsplit-final
run-unsplit-final: | $(docker_build_file)
	$(docker_run) \
		--name learning_unsplit_final \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_final

.PHONY: run-split-final
run-split-final: | $(docker_build_file)
	$(docker_run) \
		--name learning_split_final \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-e WORKER_LISTEN_PORT=50008 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_final --model VGG_server

.PHONY: run-split-client
run-split-client: | $(docker_build_file)
	$(docker_run) \
		--name learning_split_client \
		-e MASTER_UPSTREAM_PORT=50008 \
		--rm -it $(docker_image) \
		python3 benchmark.py --suite offload_client --model VGG_client --batch-size 64 --batches 64

.PHONY: run-split-intermediate
run-split-intermediate: | $(docker_build_file)
	$(docker_run) \
		--name learning_split_intermediate \
		-e MASTER_UPSTREAM_PORT=50008 \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_intermediate --model VGG_client

.PHONY: clean
clean:
	docker kill learning_data_source learning_unsplit_final learning_split_final learning_split_client learning_split_intermediate
	sudo rm -f $(docker_build_file)

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . -t $(docker_image)
	touch $(docker_build_file)

