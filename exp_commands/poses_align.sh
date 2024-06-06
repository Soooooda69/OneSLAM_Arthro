while getopts p:s:f: flag
do
    case ${flag} in
        p) path=${OPTARG};;
        s) save_path=${OPTARG};;
        f) full=${OPTARG};;
    esac
done
echo "$path"
echo "$full"
# evo_ape tum "$path"/dataset/poses_gt.txt "$path"/"$full"poses_pred.txt -as -v -p #> "$path"/evo_align_res.txt 
# evo_traj tum --ref "$path"/dataset/poses_gt.txt "$path"/"$full"poses_pred.txt -as -p
python ./scripts/helper.py --align --root_path "$path" --traj_path "$path"/dataset/poses_gt.txt