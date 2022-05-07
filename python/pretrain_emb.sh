for f in /mnt/*.reiform_ds; do
    filename="${f##*/}"
    filename=$(basename -- "$filename")
    filename="${filename%.*}"
    echo $filename
    export PYTHONPATH=/home/ben/Code/com.reiform.mynah/python
    sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH .venv/bin/python python/impl/services/modules/core/embeddings/pretraining.py $f $filename 0
done