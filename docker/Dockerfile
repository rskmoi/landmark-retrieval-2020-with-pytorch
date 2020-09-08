FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONPATH=. PATH=~/.local/bin:$PATH
RUN apt-get update && apt-get install -y openssh-server vim git
RUN pip install albumentations==0.4.6 hydra-core==1.0.0
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN export > /etc/environment
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]