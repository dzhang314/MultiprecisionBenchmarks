#!/bin/sh

set -eux

mkdir -p include
cd include

wget https://homepages.laas.fr/mmjoldes/campary/campary_01.06.17.tar.gz
tar -zxvf campary_01.06.17.tar.gz
rm campary_01.06.17.tar.gz
mv CAMPARY/Doubles/src_cpu .
rm -rf CAMPARY
mv src_cpu campary
