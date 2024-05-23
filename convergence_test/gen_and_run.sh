#!/bin/sh

for i in *.dipf
do
fn=`basename $i .dipf`
echo "cp slurm_template $fn.slurm"
cp slurm_template $fn.slurm
echo "sed -i -e "s/{fn}/$fn/g" $fn.slurm"
sed -i -e "s/{fn}/$fn/g" $fn.slurm
mv *.slurm ../
done
