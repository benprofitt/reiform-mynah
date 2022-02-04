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
	cd api ; $(GO) test -count=1 -v ./... && rm db/mynah_local.db || true
lint:
	cd api ; golangci-lint run || true
	cd api ; gosec ./... || true
clean:
	rm mynah || true
	find . -type f -name 'mynah_local.db' -delete || true
	rm mynah.json || true
	find . -type f -name 'auth.pem' -delete || true
	rm -r tmp || true
	rm -r static || true
