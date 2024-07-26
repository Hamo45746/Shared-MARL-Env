import os
import sys
import psutil
import signal
from memory_profiler import profile
import train_autoencoder
import multiprocessing as mp
import gc

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
    # Terminate all active child processes
    active_children = mp.active_children()
    for child in active_children:
        child.terminate()
        child.join(timeout=1)

    # Ensure all child processes are terminated
    for child in active_children:
        if child.is_alive():
            os.kill(child.pid, 9)  # Force kill if still alive

    # Clean up any remaining multiprocessing resources
    mp.current_process().close()

    # Terminate any remaining processes
    terminate_process_and_children(os.getpid())
    # Clear any shared memory
    try:
        import resource
        resource.RLIMIT_NOFILE
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    except (ImportError, AttributeError):
        pass
    gc.collect()

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