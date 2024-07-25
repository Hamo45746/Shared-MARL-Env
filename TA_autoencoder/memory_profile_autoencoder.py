import os
import sys
import psutil
import signal
from memory_profiler import profile
import train_autoencoder  # Import your existing script

def terminate_child_processes(pid):
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

def signal_handler(signum, frame):
    print("Interrupted. Terminating all child processes...")
    terminate_child_processes(os.getpid())
    sys.exit(1)

@profile
def run_autoencoder_training():
    # Call the main function from your train_autoencoder.py script
    train_autoencoder.main()

if __name__ == "__main__":
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run_autoencoder_training()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        terminate_child_processes(os.getpid())
        print("All child processes terminated")