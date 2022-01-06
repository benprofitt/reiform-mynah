.PHONY: site mynah format all clean
GO=go

all: site mynah
mynah:
	cd api ; $(GO) build . && cp mynah ../
site:
	echo "TODO"
format:
	cd api ; $(GO) fmt ./...
clean:
	rm mynah
