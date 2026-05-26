#!/bin/bash
#SBATCH --job-name=FoldX_Final
#SBATCH --output=foldx_final_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

FOLDX="/home/shared/programs/foldx/foldx"

#!/bin/bash
#SBATCH --job-name=FoldX_Final
#SBATCH --output=foldx_final_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

FOLDX="/home/shared/programs/foldx/foldx"

# Lista de estructuras y sus archivos correspondientes
# Carpeta | PDB_Repair | Lista_Filtrada
declare -A ESTRUCTURAS
ESTRUCTURAS["EstructuraAlphaFold"]="AF-Q92911model_Repair.pdb:individual_list_AF.txt"
ESTRUCTURAS["Estructura1"]="7UUY_Repair.pdb:individual_list_7UUY.txt"
ESTRUCTURAS["Estructura2"]="7UUZ_Repair.pdb:individual_list_7UUZ.txt"
ESTRUCTURAS["Estructura3"]="7UV0_Repair.pdb:individual_list_7UV0.txt"

for DIR in "${!ESTRUCTURAS[@]}"; do
    IFS=":" read -r PDB LISTA <<< "${ESTRUCTURAS[$DIR]}"
    
    echo "----------------------------------------------------"
    echo "Procesando $DIR..."
    cd ~/MarceNIS/FoldX/$DIR
    
    # Copiamos la lista específica generada por el mapeo de MAFFT
    cp ~/MarceNIS/FoldX/Scripts/$LISTA ./individual_list.txt
    
    if [ -f "$PDB" ]; then
        echo "Lanzando FoldX para $DIR en segundo plano..."
        $FOLDX --command=BuildModel --pdb=$PDB --mutant-file=individual_list.txt --numberOfRuns=3 > log_foldx_${DIR}.txt &
    else
        echo "ERROR: No se encontró $PDB en $DIR"
    fi
done

# CRÍTICO: Esperar a que terminen los procesos de fondo
wait

echo "¡Todo listo, Fran!"
