#! /bin/bash

# Instalación de apache :3 y php5
echo labredesML340 | sudo apt-get install apache2 -y
echo labredesML340 | sudo apt-get install php5 -y
# Instalación de mysql
sudo debconf-set-selections <<< 'mysql-server mysql-server/root_password password labredesML340'
sudo debconf-set-selections <<< 'mysql-server mysql-server/root_password_again password labredesML340'
sudo apt-get -y install mysql-server-5.5

# crear carpeta Downloads
path = /home/labredes/Downloads
if test -d /home/labredes/Downloads
then rm -R /home/labredes/Downloads
else 
mkdir /home/labredes/Downloads
fi

#Bajar en Downloads
wget -P /home/labredes/Downloads http://sourceforge.net/projects/phpscheduleit/files/latest/download?source=files

#Descomprimir :3
sudo apt-get install unzip
unzip download?source=files
cd booked
termino = lol
while [ $termino == lol ]
do

termino = otraCosa 
#Aca verifica las extensiones y les cambia los privilegios pero me dio sueño x)
