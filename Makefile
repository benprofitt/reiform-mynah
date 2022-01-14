.PHONY: site mynah format all clean
GO=go
CD=$(shell pwd)

all: site mynah
mynah:
	PKG_CONFIG_PATH=$(CD)/python cd api ; $(GO) build . && mv mynah ../
site:
	echo "TODO"
format:
	PKG_CONFIG_PATH=$(CD)/python cd api ; $(GO) fmt ./...
clean:
	rm mynah || true
	rm mynah_local.db || true
	rm mynah.json || true
	rm -r tmp || true
