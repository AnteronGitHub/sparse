uid             := $(shell id -u)
base_image_file := $(abspath ../../.DOCKER)
py_sources      := $(shell find $(abspath .) -iname *.py)

docker_image      := split_inference
docker_build_file := .DOCKER

docker_run := docker run \
               --network host \
		       -v $(abspath ../../sparse_framework):/usr/lib/sparse_framework \
		       -v $(abspath ../../data):/data \
		       -v $(abspath .):/app

.PHONY: all
all: $(docker_build_file)

.PHONY: docker
docker: $(docker_build_file)

.PHONY: run-aio
run-aio: | $(docker_build_file)
	$(docker_run) \
		--name inference_aio \
		--rm -it $(docker_image)

.PHONY: run-data-source
run-data-source: | $(docker_build_file)
	$(docker_run) \
		--name inference_data_source \
		--rm -it $(docker_image) \
		python3 benchmark.py --suite offload_source

.PHONY: run-unsplit-final
run-unsplit-final: | $(docker_build_file)
	$(docker_run) \
		--name inference_unsplit_final \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_final --model YOLOv3

.PHONY: run-split-final
run-split-final: | $(docker_build_file)
	$(docker_run) \
		--name inference_split_final \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-e WORKER_LISTEN_PORT=50008 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_final --model YOLOv3_server

.PHONY: run-split-client
run-split-client: | $(docker_build_file)
	$(docker_run) \
		--name inference_split_client \
		-e MASTER_UPSTREAM_PORT=50008 \
		--rm -it $(docker_image) \
		python3 benchmark.py --suite offload_client --model YOLOv3_local

.PHONY: run-split-intermediate
run-split-intermediate: | $(docker_build_file)
	$(docker_run) \
		--name inference_split_intermediate \
		-e WORKER_LISTEN_ADDRESS=0.0.0.0 \
		-e MASTER_UPSTREAM_PORT=50008 \
		-d $(docker_image) \
		python3 benchmark.py --suite offload_intermediate --model YOLOv3_client

.PHONY: clean
clean:
	docker kill inference_data_source inference_unsplit_final inference_split_final inference_split_client inference_split_intermediate
	sudo rm -f $(docker_build_file)

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . -t $(docker_image)
	touch $(docker_build_file)

