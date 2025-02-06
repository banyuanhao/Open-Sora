module load slurm cuda12.4/toolkit/12.4.1 gcc/13.1.0

srun --partition=hgx --nodelist=h100-01 --gres=gpu:4 --nodes=1 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --comment="ban" --pty bash

conda activate opensora_3.11_2.5.1