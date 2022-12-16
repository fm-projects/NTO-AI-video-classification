ARG DEBIAN_FRONTEND=noninteractive
FROM lizadocker/olymp_video:2

RUN pip install ftfy regex tqdm pillow
RUN pip install git+https://github.com/openai/CLIP.git

COPY weights.pt /

WORKDIR /workspace
