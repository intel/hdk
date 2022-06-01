FROM hdk:latest

RUN apt -y update
RUN apt -y upgrade

# Add a user for GitHub Actions runner 
RUN useradd -m docker 

WORKDIR /home/docker
RUN mkdir actions-runner && cd actions-runner

# Download the latest runner package
RUN curl -o actions-runner-linux-x64-2.287.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.287.1/actions-runner-linux-x64-2.287.1.tar.gz

# Validate the hash
RUN echo "8fa64384d6fdb764797503cf9885e01273179079cf837bfc2b298b1a8fd01d52  actions-runner-linux-x64-2.287.1.tar.gz" | shasum -a 256 -c

# Extract the installer
RUN tar xzf ./actions-runner-linux-x64-2.287.1.tar.gz

RUN chown -R docker:docker /home/docker
#RUN /bin/sh /home/docker/actions-runner/bin/installdependencies.sh

USER docker

# Maven Config (for proxy if required)
ARG MAVEN_CONFIG=""
RUN mkdir -p /home/docker/.m2 
RUN echo $MAVEN_CONFIG > /home/docker/.m2/settings.xml

ARG REPO=''
ARG TOKEN=''

RUN ./config.sh --unattended --url ${REPO} --token ${TOKEN}

CMD ./run.sh
