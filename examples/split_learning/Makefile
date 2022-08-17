.PHONY: docker-unsplit
docker-unsplit:
	docker build . -t unsplit_learning -f Dockerfile.unsplit

.PHONY: docker-split-server
docker-split-server:
	docker build . -t split_learning:server -f Dockerfile.server

.PHONY: docker-split-client
docker-split-client:
	docker build . -t split_learning:client -f Dockerfile.client

.PHONY: run-docker-unsplit
run-docker-unsplit:
	docker run -v $(abspath ../../src):/usr/lib/sparse -v $(abspath .):/app --rm -it unsplit_learning

.PHONY: run-docker-split-server
run-docker-split-server:
	docker run -v $(abspath ../../src):/usr/lib/sparse -v $(abspath .):/app --rm -it split_learning:server
