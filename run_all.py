import multiprocessing
import subprocess
import time
import sys

# ================= 配置区域 =================
AVAILABLE_GPUS = [0, 1, 2, 3]  # 可用的GPU ID列表

# 任务列表
DATASETS_TO_RUN = [
    "DS/code/train00.py FlakeFlagger",
    "DS/code/train10.py FlakeFlagger",
    "DS/code/train01.py FlakeFlagger",
    "DS/code/train11.py FlakeFlagger",
    "DS/code/train11.py IDoFT",
    "llama/code/train00.py FlakeFlagger",
    "llama/code/train01.py FlakeFlagger",
    "llama/code/train10.py FlakeFlagger",
    "llama/code/train11.py FlakeFlagger",
    "llama/code/train10.py IDoFT",
    "llama/code/train11.py IDoFT",
    "Qwen/code/train00.py FlakeFlagger",
    "Qwen/code/train01.py FlakeFlagger",
    "Qwen/code/train10.py FlakeFlagger",
    "Qwen/code/train11.py FlakeFlagger"
]
# ===========================================

def execute_task(cmd_str, gpu_id):
    """
    执行任务的函数
    """
    time.sleep(10)
    full_command = f"python {cmd_str} {gpu_id}"
    
    task_name = f"[{cmd_str}] on GPU {gpu_id}"
    print(f"🚀 [启动] {task_name}")

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            print(f"✅ [完成] {task_name}")
            return f"{task_name}: Success"
        else:
            print(f"❌ [失败] {task_name}")
            log_filename = f"error_{cmd_str.replace('/', '_').replace(' ', '_')}.log"
            with open(log_filename, "w") as f:
                f.write(result.stderr)
            return f"{task_name}: Failed (See {log_filename})"

    except Exception as e:
        return f"{task_name}: Exception {str(e)}"

def main():
    print(f"准备执行 {len(DATASETS_TO_RUN)} 个任务")
    print(f"可用 GPU: {AVAILABLE_GPUS}")
    
    # 准备任务参数列表
    tasks_with_args = []
    
    for i, cmd_str in enumerate(DATASETS_TO_RUN):
        target_gpu = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        tasks_with_args.append((cmd_str, target_gpu))

    pool_size = len(DATASETS_TO_RUN)
    print(f"正在启动所有 {pool_size} 个进程...")

    start_time = time.time()
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.starmap(execute_task, tasks_with_args)

    print("\n" + "="*30)
    print("所有并发任务结束")
    print(f"总耗时: {time.time() - start_time:.2f}s")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()