#!/bin/bash
#SBATCH --job-name=FoldX_Repair
#SBATCH --output=foldx_repair_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

FOLDX="/home/shared/programs/foldx/foldx"

# Copiamos los PDBs originales a sus carpetas por si no están
cp ~/MarceNIS/EstructurasNIS/7UUY.pdb ~/MarceNIS/FoldX/Estructura1/
cp ~/MarceNIS/EstructurasNIS/7UUZ.pdb ~/MarceNIS/FoldX/Estructura2/
cp ~/MarceNIS/EstructurasNIS/7UV0.pdb ~/MarceNIS/FoldX/Estructura3/
cp ~/MarceNIS/EstructurasNIS/AF-Q92911model.pdb ~/MarceNIS/FoldX/EstructuraAlphaFold/

declare -A ESTRUCTURAS
ESTRUCTURAS["Estructura1"]="7UUY.pdb"
ESTRUCTURAS["Estructura2"]="7UUZ.pdb"
ESTRUCTURAS["Estructura3"]="7UV0.pdb"
ESTRUCTURAS["EstructuraAlphaFold"]="AF-Q92911model.pdb"

for DIR in "${!ESTRUCTURAS[@]}"; do
    PDB=${ESTRUCTURAS[$DIR]}
    echo "Reparando $PDB en $DIR..."
    cd ~/MarceNIS/FoldX/$DIR
    $FOLDX --command=RepairPDB --pdb=$PDB
done

echo "¡Reparación completada!"
