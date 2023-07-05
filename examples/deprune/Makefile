base_image_file := $(abspath ../../.DOCKER)
py_sources      := $(shell find $(abspath .) -iname '*.py')

docker_image      := sparse/deprune
docker_build_file := .DOCKER

$(docker_build_file): $(py_sources) $(base_image_file)
	docker build . -t $(docker_image)
	touch $(docker_build_file)

.PHONY: docker clean

docker: $(docker_build_file)

clean:
	sudo rm -f $(docker_build_file)

