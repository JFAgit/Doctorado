#!/bin/bash

echo "--- 1. Mapeando mutaciones con los nuevos offsets ---"
python3 mapear_ratones.py

echo "--- 2. Limpiando archivos temporales de FoldX ---"
for i in 1 2 3; do
    rm -f ~/MarceNIS/FoldX/Estructura$i/molecules
    rm -f ~/MarceNIS/FoldX/Estructura$i/output
    rm -f ~/MarceNIS/FoldX/Estructura$i/*.fxout
done

echo "--- 3. Lanzando el trabajo a Slurm ---"
# Usamos el script de ratón que creamos antes
sbatch run_raton.sh

echo "--- ¡Listo! Monitoreá con: tail -f foldx_raton_*.log ---"
