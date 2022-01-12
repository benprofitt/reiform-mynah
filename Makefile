.PHONY: site mynah format all clean
GO=go

all: site mynah
mynah:
	cd api ; $(GO) build . && mv mynah ../
site:
	echo "TODO"
format:
	cd api ; $(GO) fmt ./...
clean:
	rm mynah || true
	rm mynah_local.db || true
	rm mynah.json || true
