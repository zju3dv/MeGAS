MESH_FILE="../assets/Chair.obj"
INIT_STATE="ground"


cd data_process

python 0_align_coord.py --mesh_file "$MESH_FILE" \
                        --init_state  "$INIT_STATE"

python 1_fill_internal.py --mesh_file "$MESH_FILE" \
                          --init_state  "$INIT_STATE"