help:
	@cat Makefile

DATA?="${HOME}/Data"
SRC?=$(shell dirname `pwd`)

build:
	docker build -t pybt -f docker/Dockerfile .

bash: build
	docker run -it -v $(SRC):/src/workspace -v $(DATA):/data pybt bash

ipython: build
	docker run -it -v $(SRC):/src/workspace -v $(DATA):/data pybt ipython

notebook: build
	docker run -it  -p 8888:8888 -v $(SRC):/src/workspace -v $(DATA):/data pybt

docs: build
	docker run -it  -p 8000:8000 -v $(SRC):/src/workspace -v $(DATA):/data pybt mkdocs serve --dev-addr 0.0.0.0:8000

.PHONY: build bash ipython notebook

