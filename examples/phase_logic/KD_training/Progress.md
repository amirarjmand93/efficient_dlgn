we are testing student arch . best acc is 55.5%.
 
 # our 61.7% accurayc model 
 python -m examples.phase_logic.KD_training.train_kd --mode kd --student-model tinyconv_q2_lbp2_dense2 --device cuda --teacher-backend cuda --epochs 500 --batch-size 128 --learning-rate 0.01 --seed 42 --alpha 0.5 --temperature 2.0 --student-tau 60
 # our new wide model [testing]
 python -m examples.phase_logic.KD_training.train_kd \
  --mode kd \
  --student-model tinyconv_q2_lbp2_wide10 \
  --device cuda \
  --teacher-backend cuda \
  --epochs 500 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --seed 42 \
  --alpha 0.5 \
  --temperature 2.0 \
  --student-tau 60

# The tinyconv_dense2_hls4ml/ folder is currently missing. Generate the JSON first:

  env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.student_hls4ml_exporter \
  --checkpoint \
  examples/phase_logic/KD_training/runs/cifar10_tinyconv_q2_lbp2_dense2_kd_e500_b128_lr0.02_c16_tau40_seed42_a0.5_T2_dsfreeze_20261029T031327_251150Z/best_student.pt \
  --output \
  examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls4ml_lut6_ir.json

# Then rerun your bridge command:

  env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.hls4ml_bridge.convert_json \
  --json examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls4ml_lut6_ir.json \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls_project_v2 \
  --backend Vitis \
  --part xczu7ev-ffvc1156-2-e \
  --write