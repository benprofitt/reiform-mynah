.PHONY: frontend mynah format test clean all
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
clean:
	rm mynah || true
	rm mynah_local.db || true
	rm mynah.json || true
	rm auth.pem || true
	rm -r tmp || true
	rm -r static || true
