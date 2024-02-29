start:
	docker run -d \
	-it \
	--name pmm \
	-p 51351:51351 \
	--workdir /opt \
	--gpus all \
	-v /home/users/Documents/workspace/pmm:/opt/pmm \
	pytorchlightning/base-cuda-py3.10-torch2.1-cuda12.1.0 \
	bash
	
	docker exec -it pmm bash -c "pip install -r pmm/Dockerfile/requirements.txt"

down:
	docker stop pmm
	docker rm pmm

access:
	docker exec -it pmm bash
