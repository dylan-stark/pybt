help:
	@cat Makefile

DATA?="${HOME}/Data"
SRC?=$(shell dirname `pwd`)

build:
	docker build -t pybt .

bash: build
	docker run -it -v $(SRC):/src/workspace -v $(DATA):/data pybt bash

ipython: build
	docker run -it -v $(SRC):/src/workspace -v $(DATA):/data pybt ipython

notebook: build
	docker run -it  -p 8888:8888 -v $(SRC):/src/workspace -v $(DATA):/data pybt

.PHONY: build bash ipython notebook

