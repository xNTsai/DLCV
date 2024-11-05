# 3 models
for filename in "0013_sat" "0062_sat" "0104_sat"
do
    for folder in "early" "middle" "final"
    do
        python3 viz_mask.py --img_path="./test_img/$filename.jpg" --seg_path="./mask_$folder/$filename.png"
        mv exp.png "./mask_$folder/$filename rst.png"
    done
done
