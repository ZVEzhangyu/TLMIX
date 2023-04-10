#!/bin/bash
trap 'onCtrlC' INT

function onCtrlC () {
    echo 'Ctrl+C is captured'
    for pid in $(jobs -p); do
        kill -9 $pid
        echo $pid
    done

    kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
    exit 1
}
# maps=1c3s5z,2c_vs_64zg,2m_vs_1z,2s_vs_1sc,2s3z,3m,3s_vs_3z,3s_vs_4z,3s_vs_5z,3s5z_vs_3s6z,3s5z,5m_vs_6m,6h_vs_8z,8m_vs_9m,8m,10m_vs_11m,25m,27m_vs_30m,bane_vs_bane,corridor,MMM,MMM2,so_many_baneling
maps=stag_hunt
algos=tlmix,iql,qtran,coma
maps=(${maps//,/ })
algos=(${algos//,/ })

if [ ! $maps ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)"
    exit 1
fi

echo "MAP LIST:" ${maps[@]}

for algo in "${algos[@]}"; do
    for((i=0;i<5;i++)); do
        for map in "${maps[@]}"; do
            # CUDA_VISIBLE_DEVICES="0" python3 src/main.py --config="$algo" --env-config="sc2" with env_args.map_name="$map" &
            CUDA_VISIBLE_DEVICES="0" python3 src/main.py --config="$algo" --env-config="stag_hunt" with env_args.map_name="$map" &
            wait
            sleep $((RANDOM % 2 + 20))
        done 
    done
done
wait
