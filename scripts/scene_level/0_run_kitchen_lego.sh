BASE_DIR="360_kitchen"

# 通用参数
THERMAL_BOUNDARY="sphere"
THERMAL_SPHERE_BOUNDARY="${BASE_DIR}.txt"
THERMAL_BOUNDARY_VALUE="0.5"
INCREMENTAL_THERMAL_BC=true  # 若要启用增量热边界条件，将此改为 true
SOLID_MATERIAL="jelly_smooth" # "jelly"
LIQUID_MATERIAL="foam_smooth" #"foam"
SAVE_NOTE="damping_soft_vis100"

USE_DAMPING=true   # 若要开启阻尼，将此改为 true

GS_TYPE="ours"

echo "Running simulation for: $GS_FILE ($GS_TYPE)"

CMD="python gs_simulation.py \
        --gs_file $BASE_DIR \
        --gs_type $GS_TYPE \
        --solid_material $SOLID_MATERIAL \
        --liquid_material $LIQUID_MATERIAL \
        --thermal_boundary $THERMAL_BOUNDARY \
        --thermal_sphere_boundary $THERMAL_SPHERE_BOUNDARY \
        --thermal_boundary_value $THERMAL_BOUNDARY_VALUE \
        --save_note $SAVE_NOTE"

# 若启用 damping
if [ "$USE_DAMPING" = true ]; then
  CMD="$CMD --use_damping"
fi
    # 若启用增量热边界条件
if [ "$INCREMENTAL_THERMAL_BC" = true ]; then
  CMD="$CMD --incremental_thermal_bc"
fi

echo "Executing: $CMD"
eval $CMD
echo "--------------------------------------"