cd ./curation/Sample_Zero-Shot_Grounding_RSNA || exit

# --data_type: caption, vqa
# --dataset_name: mimic, slake, vqa-rad, iuxray
python ./inference_attention-map_score.py \
    --config ./MedKLIP_config.yaml \
    --model_path /home/cormac/MMedPO/checkpoint_medklip.pth \
    --dataset_name slake \
    --dataset_type caption \
    --image_root /home/cormac/MMedPO/SLAKE/imgs \
    --annotation_save_root /home/cormac/MMedPO/dataset/annotation \
    --noised_image_save_root /home/cormac/MMedPO/dataset/noised_image \
    --gpu 0

