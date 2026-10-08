BASE_DIR="sculpture"

# 通用参数
THERMAL_BOUNDARY="sphere"
THERMAL_SPHERE_BOUNDARY="${BASE_DIR}.txt"
THERMAL_BOUNDARY_VALUE="0.5"

SAVE_NOTE="test"
USE_DAMPING=true   # 若要开启阻尼，将此改为 true
INCREMENTAL_THERMAL_BC=true  # 若要启用增量热边界条件，将此改为 true

# 定义不同场景的文件与类型
declare -A GS_CONFIGS=(
    ["ours"]="ours"
    ["physgaussian"]="3dgs"
    ["vrdoh"]="3dgs"
    ["3dgs"]="3dgs"
)

# 遍历执行
ORDER=("ours" "physgaussian" "vrdoh" "3dgs")
for NAME in "${ORDER[@]}"; do
    GS_TYPE="${GS_CONFIGS[$NAME]}"
    GS_FILE="${BASE_DIR}/${NAME}"

    echo "Running simulation for: $GS_FILE ($GS_TYPE)"

    CMD="python gs_simulation.py \
        --gs_file $GS_FILE \
        --gs_type $GS_TYPE \
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
done