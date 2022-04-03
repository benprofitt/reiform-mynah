.PHONY: frontend mynah format test data-test clean all lint
GO=/usr/local/go/bin/go
GO_ARGS= #-tags s3

all: frontend mynah
mynah:
	cd api ; $(GO) build $(GO_ARGS) . && mv mynah ../
frontend:
	rm -r static || true
	cd frontend ; npm install && npm run build && mv build ../static
format:
	cd api ; $(GO) fmt ./...
test:
	cd api ; $(GO) test -count=1 -timeout 30s -v ./...
data-test:
	PYTHONPATH=$(PYTHONPATH):$(PWD)/python python3.7 python/impl/test/image_classification/mislabeled_tests.py
lint:
	cd api ; golangci-lint run || true
	cd api ; gosec ./... || true
	mypy --config-file .mypy.ini . || true
clean:
	-@rm -r python/__pycache__ 2> /dev/null || true
	-@find api/ -type f -name 'mynah_local.db' -delete || true
	-@find api/ -type f -name 'auth.pem' -delete || true
	-@rm mynah 2> /dev/null || true
	-@rm mynah_local.db 2> /dev/null || true
	-@rm mynah.json 2> /dev/null || true
	-@rm auth.pem 2> /dev/null || true
	-@rm -r data 2> /dev/null || true
	-@rm -r static 2> /dev/null || true
