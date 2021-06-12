# Homework_medicalimage
run train.py to train the model
python -m train --train_image_dir <> --train_label_dir <> --validate_image_dir <> --validate_label_dir <>
run predict.py to generate the prediction images
python -m predict --image_dir <>--pre_dir <> --model_path <>
run evaluation.py to evaluate
python -m evaluation --gt_dir <> --pred_dir <> --clf False
