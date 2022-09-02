# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"
  config.vm.hostname = "MynahDev"

  config.vm.provider "virtualbox" do |v|
    v.memory = 8192
    v.cpus = 4
  end

  config.disksize.size = '18GB'

  config.vm.provision "shell", inline: <<-SHELL
		sudo apt-get update
		sudo apt-get -y upgrade
		sudo apt-get install -y valgrind
		sudo apt-get install -y make
		apt-get install -y git
		apt-get install -y curl wget
		wget https://golang.org/dl/go1.18.linux-amd64.tar.gz
		rm -rf /usr/local/go && tar -C /usr/local -xzf go1.18.linux-amd64.tar.gz
		sudo apt-get install -y python3.8
        sudo apt-get install -y python3-pip
        echo 'python deps installation start'
        pushd /vagrant
        ./python/py_install_cpu.sh
        popd
        curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
        sudo apt install -y nodejs
        curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b /home/vagrant/go/bin v1.44.0
        curl -sfL https://raw.githubusercontent.com/securego/gosec/master/install.sh | sh -s -- -b /home/vagrant/go/bin v2.9.6
        echo 'export PATH=$PATH:/usr/local/go/bin' >> /home/vagrant/.bashrc
        echo 'export PATH=$PATH:/home/vagrant/go/bin' >> /home/vagrant/.bashrc
        echo 'export PKG_CONFIG_PATH=/vagrant/python' >> /home/vagrant/.bashrc
        echo 'source /usr/share/bash-completion/completions/git' >> /home/vagrant/.bashrc
  SHELL

end
