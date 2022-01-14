# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.hostname = "MynahDev"

  config.vm.provider "virtualbox" do |v|
    v.memory = 8192
    v.cpus = 4
  end

  config.vm.provision "shell", inline: <<-SHELL
		sudo apt-get update
		sudo apt-get -y upgrade
		sudo apt-get install -y valgrind
		sudo apt-get install -y make
		apt-get install -y git
		apt-get install -y curl wget
		wget https://golang.org/dl/go1.17.6.linux-amd64.tar.gz
		rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.6.linux-amd64.tar.gz
		sudo apt-get install -y python3.7-dev
		sudo apt-get install -y python3.7
		sudo apt-get install -y pkg-config
    sudo apt-get install -y python3-pip
    python3.7 -m pip install Cython
  SHELL

end
