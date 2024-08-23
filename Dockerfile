ARG repository=ccr.ccs.tencentyun.com/library
ARG version=latest
FROM ${repository}/ubuntu:${version}
LABEL authors="Clothoid"
WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]