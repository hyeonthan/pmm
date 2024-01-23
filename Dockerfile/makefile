start:
	docker run -d \
	-it \
	--name service \
	-p 51351:51351 \
	--workdir /opt \
	--gpus all \
	-v /home/users/Documents/workspace/pmm:/opt/service \
	pytorchlightning/base-cuda-py3.10-torch2.1-cuda12.1.0 \
	bash
	
	docker exec -it service bash -c "pip install -r service/Dockerfile/requirements.txt"

down:
	docker stop service
	docker rm service

access:
	docker exec -it service bash