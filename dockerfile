FROM pytorch/pytorch:latest

CMD apt update && apt install -y git
CMD git clone https://github.com/authoranonymous321/DA.git && cd DA && python3.8 -m pip install -e .
# TODO DPP training: chmod 777 /dev/shm && chmod 777 /dev/shm/*
