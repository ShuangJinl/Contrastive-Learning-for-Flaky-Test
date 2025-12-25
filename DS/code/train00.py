import os

test_only = False
projects = False
 
# ====================================

# 2. 路径配置
if projects:
    fix = "_projects"
else:
    fix = ""
llama_model_path = "/data/public/deepseek-coder-6.7b-instruct"
train_pair_path = f"../../new_dataset{fix}/IDoFT/flaky_train_s.csv"
test_pair_path = f"../../new_dataset{fix}/IDoFT/flaky_test.csv"
code_path = f"../../new_dataset{fix}/IDoFT/flaky_db.csv"

# 3. 根据 test_only 自动决定 save_dir 和 命令行参数
if test_only:
    save_dir = f"saved_models{fix}/test_only/IDoFT/"
    test_only_arg = "--test_only" 
else:
    save_dir = f"saved_models{fix}/full/IDoFT/"
    test_only_arg = ""            

# 4. 根据 use_project_split 决定参数
if projects:
    split_arg = "--project_split"
    split_note = "Project-Level Split"
else:
    split_arg = ""
    split_note = "Random Split"

# 5. 这里修改 GPU 编号
GPU_ID = "1" 

# ===========================================

common_args = (
    f"--llama_model_path={llama_model_path} "
    f"--train_pair_path={train_pair_path} "
    f"--test_pair_path={test_pair_path} "
    f"--code_path={code_path} "
    f"--save_dir={save_dir} "
    f"--batch_size=2 "
    f"--epochs=7 "
    f"--lr=1e-5 "
    f"{test_only_arg} "
    f"{split_arg}" # 传递划分参数
)

print(f"==================================================")
print(f"Global Configuration:")
print(f"  Test Only Mode: {test_only}")
print(f"  Split Strategy: {split_note}")
print(f"==================================================\n")

# ---------------------------------------------------------
# 1. 运行 SFT (Pair)
# ---------------------------------------------------------
print(f">>> Running SFT... [{split_note}]")
cmd_sft = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run.py {common_args} --training_mode=sft"
os.system(cmd_sft)

# ---------------------------------------------------------
# 2. 运行 Contrastive Only (Mix Loss)
# ---------------------------------------------------------
print(f"\n>>> Running Contrastive (Mix Loss)... [{split_note}]")
cmd_con = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run.py {common_args} --training_mode=con"
os.system(cmd_con)

# ---------------------------------------------------------
# 3. 运行 Im_Con (Improved Contrastive)
# ---------------------------------------------------------
print(f"\n>>> Running Im_Con... [{split_note}]")
cmd_im_con = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run.py {common_args} --training_mode=im_con"
os.system(cmd_im_con)

# ---------------------------------------------------------
# 4. 运行 Full CL (Pure Contrastive Loss) [NEW]
# ---------------------------------------------------------
print(f"\n>>> Running Full CL (Pure CL)... [{split_note}]")
cmd_full_con = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run.py {common_args} --training_mode=full_con"
os.system(cmd_full_con)

# ---------------------------------------------------------
# 5. 运行 CL then SFT (Two Stage) [NEW]
# ---------------------------------------------------------
print(f"\n>>> Running CL then SFT (Two Stage)... [{split_note}]")
cmd_cl_sft = f"CUDA_VISIBLE_DEVICES={GPU_ID} python run.py {common_args} --training_mode=cl_sft"
os.system(cmd_cl_sft)