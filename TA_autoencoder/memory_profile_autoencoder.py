import os
import sys
import psutil
import signal
from memory_profiler import profile
import train_autoencoder
import multiprocessing as mp

def terminate_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        gone, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        if parent.is_running():
            parent.terminate()
            parent.wait(3)
    except psutil.NoSuchProcess:
        pass

def cleanup_resources():
    # Terminate all child processes
    for child in mp.active_children():
        child.terminate()
        child.join(timeout=1)
    
    # Force cleanup of any remaining resources
    mp.util.cleanup_remaining_resources()
    
    # Terminate any remaining processes
    terminate_process_and_children(os.getpid())

def signal_handler(signum, frame):
    print("Interrupted. Cleaning up resources...")
    cleanup_resources()
    sys.exit(1)

@profile
def run_autoencoder_training():
    # Ensure the main function uses a method that allows for clean interruption
    train_autoencoder.main()

if __name__ == "__main__":
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        mp.set_start_method('spawn', force=True)  # Use 'spawn' method for better interrupt handling
        run_autoencoder_training()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        cleanup_resources()
        print("All resources cleaned up")