FROM hdk:latest

RUN apt -y update
RUN apt -y upgrade

# Add a user for GitHub Actions runner 
RUN useradd -m docker 

WORKDIR /home/docker
RUN mkdir actions-runner && cd actions-runner

# Download the latest runner package
RUN curl -o actions-runner-linux-x64-2.292.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.292.0/actions-runner-linux-x64-2.292.0.tar.gz

# Validate the hash
RUN echo "14839ba9f3da01abdd51d1eae0eb53397736473542a8fae4b7618d05a5af7bb5  actions-runner-linux-x64-2.292.0.tar.gz" | shasum -a 256 -c

# Extract the installer
RUN tar xzf ./actions-runner-linux-x64-2.292.0.tar.gz

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
