D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train.py \
--data_star_num 1 --data_end_num 1000 \
--maskname poisson2d --maskperc  5  \
--model_name  poisson2d_5_v.t7  --model_log  log_possion2d_5_v \
--epochs 50 --lr 1e-3\

D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train.py \
--data_star_num 1 --data_end_num 1000 \
--maskname poisson2d --maskperc  5  \
--model_name  poisson2d_5_v0.t7  --model_log  log_possion2d_5_v0 \
--epochs 80 --lr 1e-3\

D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train.py \
--data_star_num 1 --data_end_num 1000 \
--maskname poisson2d --maskperc  1  \
--model_name  poisson2d_1_v0.t7  --model_log  log_possion2d_1_v0 \
--epochs 80 --lr 1e-3\

D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train.py \
--data_star_num 1 --data_end_num 1000 \
--maskname gaussian2d --maskperc  1  \
--model_name  gaussian2d_1_v0.t7  --model_log  log_gaussian2d_1_v0 \
--epochs 80 --lr 1e-3\

D:/anconda3/python.exe -u  G:/lxb/desktop/easyCS/train.py \
--data_star_num 1 --data_end_num 1000 \
--maskname gaussian2d --maskperc  5  \
--model_name  gaussian2d_5_v0.t7  --model_log  log_gaussian2d_5_v0 \
--epochs 80 --lr 1e-3\


