.PHONY: frontend api format test data-test clean all lint cli setup
GO=/usr/local/go/bin/go
GO_ARGS= #-tags s3

all: frontend api
api:
	cd api ; $(GO) build $(GO_ARGS) . && mv mynah ../
cli:
	cd cli ; $(GO) build $(GO_ARGS) . && mv mynah-cli ../
frontend:
	rm -r static || true
	cd frontend ; npm install && npm run build && mv dist ../static
format:
	cd api ; $(GO) fmt ./...
test:
	cd api ; $(GO) test -count=1 -timeout 30s -v ./...
	cd cli ; $(GO) test -count=1 -timeout 30s -v ./...
data-test:
	PYTHONPATH=$(PYTHONPATH):$(PWD)/python python3.8 python/impl/test/image_classification/mislabeled_tests.py
lint:
	cd api ; golangci-lint run || true
	cd api ; gosec ./... || true
	cd cli ; golangci-lint run || true
	cd cli ; gosec ./... || true
	mypy --config-file .mypy.ini . || true
clean:
	-@rm -r python/__pycache__ 2> /dev/null || true
	-@find api/ -type f -name 'mynah_local.db' -delete || true
	-@find api/ -type f -name 'auth.pem' -delete || true
	-@rm mynah 2> /dev/null || true
	-@rm mynah-cli 2> /dev/null || true
	-@rm -r data 2> /dev/null || true
	-@rm -r static 2> /dev/null || true
