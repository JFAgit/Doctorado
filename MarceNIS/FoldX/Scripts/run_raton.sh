#!/bin/bash
#SBATCH --job-name=FoldX_Raton
#SBATCH --output=/home/faguilera/MarceNIS/FoldX/foldx_raton_%j.log
#SBATCH --error=/home/faguilera/MarceNIS/FoldX/foldx_raton_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

FOLDX="/home/shared/programs/foldx/foldx"

echo "--- Iniciando proceso de Ratón: $(date) ---"

# Estructura 1
echo "Procesando Estructura 1 (7UUY)..."
cd /home/faguilera/MarceNIS/FoldX/Estructura1
$FOLDX --command=BuildModel --pdb=7UUY_Repair.pdb --mutant-file=individual_list.txt --numberOfRuns=1

# Estructura 2
echo "Procesando Estructura 2 (7UUZ)..."
cd /home/faguilera/MarceNIS/FoldX/Estructura2
$FOLDX --command=BuildModel --pdb=7UUZ_Repair.pdb --mutant-file=individual_list.txt --numberOfRuns=1

# Estructura 3
echo "Procesando Estructura 3 (7UV0)..."
cd /home/faguilera/MarceNIS/FoldX/Estructura3
$FOLDX --command=BuildModel --pdb=7UV0_Repair.pdb --mutant-file=individual_list.txt --numberOfRuns=1

echo "--- Proceso finalizado: $(date) ---"
