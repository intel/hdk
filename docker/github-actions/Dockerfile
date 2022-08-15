FROM hdk:latest

RUN apt -y update
RUN apt -y upgrade

# Add a user for GitHub Actions runner 
RUN useradd -m docker 

WORKDIR /home/docker
RUN mkdir actions-runner && cd actions-runner

# Download the latest runner package
RUN curl -o actions-runner-linux-x64-2.295.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.295.0/actions-runner-linux-x64-2.295.0.tar.gz

# Validate the hash
RUN echo "a80c1ab58be3cd4920ac2e51948723af33c2248b434a8a20bd9b3891ca4000b6  actions-runner-linux-x64-2.295.0.tar.gz" | shasum -a 256 -c

# Extract the installer
RUN tar xzf ./actions-runner-linux-x64-2.295.0.tar.gz

RUN chown -R docker:docker /home/docker

USER docker

# Maven Config (for proxy if required)
ARG MAVEN_CONFIG=""
RUN mkdir -p /home/docker/.m2 
RUN echo $MAVEN_CONFIG > /home/docker/.m2/settings.xml

ARG REPO=''
ARG TOKEN=''

RUN ./config.sh --unattended --url ${REPO} --token ${TOKEN}

CMD ./run.sh
