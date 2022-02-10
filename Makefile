.PHONY: frontend mynah format test clean all lint
GO=go

all: frontend mynah
mynah:
	cd api ; $(GO) build . && mv mynah ../
frontend:
	rm -r static || true
	cd frontend ; npm install && npm run build && mv build ../static
format:
	cd api ; $(GO) fmt ./...
test:
	cd api ; $(GO) test -count=1 -timeout 30s -v ./... || true
lint:
	cd api ; golangci-lint run || true
	cd api ; gosec ./... || true
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
