FROM ubuntu:16.04
MAINTAINER Van Quang Nguyen
RUN apt-get update
RUN cat /etc/lsb-release
RUN apt-get install -y python3-pip
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib seaborn sklearn
RUN mkdir -p /cebd1160/
COPY ./pscript.py /cebd1160/pscript.py
ENTRYPOINT ["python3", "/cebd1160/pscript.py"]
