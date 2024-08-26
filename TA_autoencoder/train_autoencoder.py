import os
import sys
import yaml
import multiprocessing as mp
from tqdm import tqdm
import logging
import psutil
import time
import gc
import signal
import fcntl
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from multiprocessing import Pool, Value, cpu_count
import traceback
# from test_autoencoder import test_specific_autoencoder

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from autoencoder import EnvironmentAutoencoder

# Constants
H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
H5_PROGRESS_FILE = 'h5_collection_progress.txt'
AUTOENCODER_FILE = 'trained_autoencoder.pth'
TRAINING_STATE_FILE = 'training_state.pth'
STEPS_PER_EPISODE = 100
LOG_FILE = 'autoencoder_training.log'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Complete path to the log file in the Shared-MARL-Env folder
log_file_path = os.path.join(parent_dir, LOG_FILE)

# Global flag to indicate interruption
interrupt_flag = Value('i', 0)
temp_flag = None

def setup_logging():
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Remove any existing handlers to avoid duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up file logging
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Test logging
        logging.info(f"Logging initialized. Log file: {log_file_path}")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        print(f"Will log to console only.")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
def log_error(depth=2):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = traceback.extract_tb(exc_traceback)
    
    # Filter to only include frames from files in the current directory or its subdirectories
    current_dir = os.path.abspath(os.getcwd())
    relevant_frames = [frame for frame in tb if os.path.commonpath([current_dir, frame.filename]) == current_dir]
    
    # Take the outermost 'depth' number of frames
    outer_frames = relevant_frames[:depth]
    
    error_msgs = []
    for frame in reversed(outer_frames):  # Reverse to show outermost first
        error_msg = f"Error in file '{frame.filename}', line {frame.lineno}, in {frame.name}"
        error_msgs.append(error_msg)
    
    error_msg = " | ".join(error_msgs) + f": {exc_type.__name__}: {exc_value}"

    logging.error(error_msg)
    return error_msg

class FlattenedMultiAgentH5Dataset(Dataset):
    def __init__(self, h5_files, dtype=torch.float32):
        self.h5_files = h5_files
        self.file_indices = []
        self.step_indices = []
        self.agent_indices = []
        self.cumulative_lengths = [0]
        self.dtype = dtype
        
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                if 'data' not in f:
                    logging.warning(f"No 'data' group in file {h5_file}")
                    continue
                data = f['data']
                for step_idx in data.keys():
                    step_data = data[step_idx]
                    for agent_idx in step_data.keys():
                        self.file_indices.append(file_idx)
                        self.step_indices.append(step_idx)
                        self.agent_indices.append(agent_idx)
            
            self.cumulative_lengths.append(len(self.file_indices))
        
        self.total_length = len(self.file_indices)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        step_idx = self.step_indices[idx]
        agent_idx = self.agent_indices[idx]
        
        with h5py.File(self.h5_files[file_idx], 'r') as f:
            data = f['data']
            step_data = data[step_idx]
            agent_data = step_data[agent_idx]
            full_state = agent_data['full_state'][()]
            
            # Convert to torch tensor and ensure it's float32
            full_state_tensor = torch.from_numpy(full_state).to(self.dtype)
            
            # Ensure we have all 4 layers
            if full_state_tensor.shape[0] != 4:
                raise ValueError(f"Expected 4 layers, but got {full_state_tensor.shape[0]} layers in file {self.h5_files[file_idx]}, step {step_idx}, agent {agent_idx}")
            
            # Create a dictionary with each layer as a separate item
            return {f'layer_{i}': full_state_tensor[i] for i in range(full_state_tensor.shape[0])}

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            return max(temp.current for temp in temps['coretemp'])
        return None
    except:
        return None

def init_worker():
    global interrupt_flag
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def init_worker_with_temp_flag(flag):
    global temp_flag
    temp_flag = flag

def signal_handler(signum, frame):
    global interrupt_flag
    print("Interrupt received, stopping processes...")
    if interrupt_flag is not None:
        interrupt_flag.value = 1

def cleanup_resources():
    active_children = mp.active_children()
    for child in active_children:
        child.terminate()
        child.join(timeout=1)

    for child in active_children:
        if child.is_alive():
            os.kill(child.pid, 9)  # Force kill if still alive

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
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
        
    gc.collect()
    
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        import resource
        resource.RLIMIT_NOFILE
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    except (ImportError, AttributeError):
        pass

    mp.current_process().close()
    gc.collect()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def is_dataset_complete(filepath, steps_per_episode):
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'data' not in hf:
                logging.warning(f"'data' group missing in {filepath}")
                return False
            actual_steps = len(hf['data'])
            if actual_steps != steps_per_episode:
                logging.warning(f"Incomplete dataset in {filepath}: {actual_steps}/{steps_per_episode} steps")
                return False
            return True
    except Exception as e:
        logging.error(f"Error checking dataset completeness for {filepath}: {str(e)}")
        return False

def acquire_lock(lock_file):
    while True:
        try:
            lock_fd = open(lock_file, 'w')
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except IOError:
            time.sleep(0.1)

def release_lock(lock_fd):
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()
    # os.remove(lock_fd)

def load_progress(all_configs):
    progress = set()
    for config in all_configs:
        map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
        filename = f"data_m{map_name}_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
        filepath = os.path.join(H5_FOLDER, filename)
        if os.path.exists(filepath) and is_dataset_complete(filepath, STEPS_PER_EPISODE):
            progress.add(filename)
    return progress

# def process_config(args):
#     config, config_path, h5_folder, steps_per_episode = args
#     lock_fd = None
#     try:
#         map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
#         filename = f"data_m{map_name}_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
#         filepath = os.path.join(h5_folder, filename)
#         lock_file = f"{filepath}.lock"

#         # Try to acquire the lock
#         lock_fd = acquire_lock(lock_file)
#         if lock_fd is None:
#             print(f"File {filename} is being processed by another worker. Skipping.")
#             return None

#         # Check if the file already exists and is complete
#         if os.path.exists(filepath) and is_dataset_complete(filepath, steps_per_episode):
#             print(f"File {filename} already exists and is complete. Skipping.")
#             return filepath

#         # If the file doesn't exist or is incomplete, collect the data
#         filepath = collect_data_for_config(config, config_path, steps_per_episode, h5_folder)

#         if is_dataset_complete(filepath, steps_per_episode):
#             # Verify that the file contains data for all 4 layers
#             with h5py.File(filepath, 'r') as f:
#                 first_step = f['data']['0']
#                 first_agent = first_step[list(first_step.keys())[0]]
#                 full_state = first_agent['full_state'][()]
#                 if full_state.shape[0] != 4:
#                     logging.error(f"Incorrect number of layers in {filepath}: expected 4, got {full_state.shape[0]}")
#                     return None
#             return filepath
#         else:
#             return None

#     except Exception as e:
#         logging.error(f"Error processing config {config}: {str(e)}")
#         log_error()
#         return None

#     finally:
#         # Always release the lock if it was acquired
#         if lock_fd is not None:
#             release_lock(lock_fd)


def process_config_wrapper(args):
    try:
        if interrupt_flag.value:
            return None
        return process_config(args)
    except KeyboardInterrupt:
        return None

def process_config(args):
    try:
        config, config_path, h5_folder, steps_per_episode = args
        filepath = collect_data_for_config(config, config_path, steps_per_episode, h5_folder)
        if is_dataset_complete(filepath, steps_per_episode):
            # Verify that the file contains data for all 4 layers
            with h5py.File(filepath, 'r') as f:
                first_step = f['data']['0']
                first_agent = first_step[list(first_step.keys())[0]]
                full_state = first_agent['full_state'][()]
                if full_state.shape[0] != 4:
                    logging.error(f"Incorrect number of layers in {filepath}: expected 4, got {full_state.shape[0]}")
                    return None
            # progress = set(os.path.basename(filepath))
            # progress.add(os.path.basename(filepath))
            # save_progress(progress)
            return filepath
        else:
            return None
    except Exception as e:
        logging.error(f"Error processing config {config}: {str(e)}")
        log_error()
        return None

def is_config_processed(config, h5_folder, steps_per_episode):
    map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
    filename = f"data_m{map_name}_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
    filepath = os.path.join(h5_folder, filename)
    return os.path.exists(filepath) and is_dataset_complete(filepath, steps_per_episode)

def save_progress(completed_configs):
    progress_file = os.path.join(H5_FOLDER, H5_PROGRESS_FILE)
    lock_file = f"{progress_file}.lock"
    lock_fd = acquire_lock(lock_file)
    try:
        with open(progress_file, 'w') as f:
            for config in completed_configs:
                f.write(f"{config}\n")
    finally:
        release_lock(lock_fd)
        os.remove(lock_file)

def collect_data_for_config(config, config_path, steps_per_episode, h5_folder):
    env = Environment(config_path)
    env.config.update(config)
    env.num_agents = config['n_agents']
    env.reset()

    # Update filename to include map information
    map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
    if config.get('generate_rand_map', False):
        map_name = 'rand'
    filename = f"data_m{map_name}_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
    filepath = os.path.join(h5_folder, filename)

    start_step = 0
    if os.path.exists(filepath):
        with h5py.File(filepath, 'r') as hf:
            if 'data' in hf:
                existing_steps = [int(step) for step in hf['data'].keys()]
                if existing_steps:
                    start_step = max(existing_steps) + 1
                    logging.info(f"Resuming data collection for {filename} from step {start_step}")
                    
                    if start_step >= steps_per_episode:
                        logging.info(f"All steps already collected for {filename}")
                        return filepath

    with h5py.File(filepath, 'a') as hf:
        if 'data' not in hf:
            dataset = hf.create_group('data', track_order=True)
        else:
            dataset = hf['data']
        
        if start_step > 0:
            for _ in range(start_step):
                action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}
                env.step(action_dict)

        for step in tqdm(range(start_step, steps_per_episode), desc=f"Collecting data for {filename}", initial=start_step, total=steps_per_episode):
            if interrupt_flag.value:
                print(f"Interrupting process for config: {config}")
                return None

            action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}
            observations, rewards, done, info = env.step(action_dict)
            
            if not isinstance(observations, dict):
                logging.error(f"Unexpected observation type: {type(observations)}")
                break

            step_group = dataset.create_group(str(step), track_order=True)
            for agent_id, obs in observations.items():
                agent_group = step_group.create_group(str(agent_id), track_order=True)
                
                agent_group.create_dataset('full_state', data=obs['full_state'], 
                                           compression="gzip", compression_opts=9)
                agent_group.create_dataset('local_obs', data=obs['local_obs'], 
                                           compression="gzip", compression_opts=9)
            
            if step % 10 == 0:
                hf.flush()
                gc.collect()
                mem_usage = psutil.virtual_memory().percent
                logging.info(f"Step {step}: Memory usage {mem_usage}%")
                
                if mem_usage > 90:
                    logging.warning(f"High memory usage detected: {mem_usage}%")
                    break
            
            if done:
                break

    logging.info(f"Data collection complete. File saved: {filepath}")
    return filepath

def save_training_state(autoencoder, ae_index, epoch):
    try:
        state = {
            'epoch': epoch,
            'model_state_dict': autoencoder.autoencoders[ae_index].cpu().state_dict(),
            'optimizer_state_dict': autoencoder.cpu_state_dict(autoencoder.optimizers[ae_index]),
            'scheduler_state_dict': autoencoder.schedulers[ae_index].state_dict(),
            'scaler': autoencoder.scaler.state_dict(),
        }
        save_path = os.path.join(H5_FOLDER, f"training_state_layer_{ae_index}.pth")
        torch.save(state, save_path)
        logging.info(f"Saved training state for autoencoder {ae_index} at epoch {epoch} to {save_path}")
        
        # Move the autoencoder back to the original device
        autoencoder.autoencoders[ae_index].to(autoencoder.device)
    except Exception as e:
        logging.error(f"Error saving training state for autoencoder {ae_index}: {str(e)}")
        logging.error(traceback.format_exc())

def load_training_state(autoencoder, ae_index):
    state_path = os.path.join(H5_FOLDER, f"training_state_layer_{ae_index}.pth")
    if os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location='cpu')
            
            autoencoder.autoencoders[ae_index].load_state_dict(state['model_state_dict'])
            autoencoder.autoencoders[ae_index].to(autoencoder.device)
            
            autoencoder.optimizers[ae_index].load_state_dict(state['optimizer_state_dict'])
            autoencoder.move_optimizer_to_device(autoencoder.optimizers[ae_index], autoencoder.device)
            
            autoencoder.schedulers[ae_index].load_state_dict(state['scheduler_state_dict'])
            
            if 'scaler' in state:
                autoencoder.scaler.load_state_dict(state['scaler'])
            
            logging.info(f"Loaded training state for autoencoder {ae_index} from {state_path}")
            return state['epoch']
        except Exception as e:
            logging.error(f"Error loading training state for autoencoder {ae_index}: {str(e)}")
            logging.error(traceback.format_exc())
            return 0
    return 0
        
def train_autoencoder(autoencoder, h5_files_low_jammers, h5_files_all_jammers, num_epochs=100, batch_size=32, patience=2, delta=0.01, load_previous=None, load_optimizer=False):
    device = autoencoder.device
    output_folder = os.path.join(H5_FOLDER, 'training_visualisations')
    os.makedirs(output_folder, exist_ok=True)
    dtype = autoencoder.dtype
    writer = SummaryWriter(log_dir=os.path.join(H5_FOLDER, 'tensorboard_logs'))

    # Set log interval
    log_interval = 100  # Log every 100 batches

    try:
        for ae_index in range(0, 3):  # Train each autoencoder separately
            print(f"Training autoencoder {ae_index}")
            if ae_index == 0:
                delta = 0.001
            # load from previous save if specified
            if load_previous and ae_index in load_previous:
                previous_save_path = os.path.join(H5_FOLDER, f"AE_save_22_08/autoencoder_{ae_index}_best.pth") # Change prev AE path here
                if os.path.exists(previous_save_path):
                    logging.info(f"Loading previous weights for autoencoder {ae_index}")
                    checkpoint = torch.load(previous_save_path, map_location=device)
                    autoencoder.autoencoders[ae_index].load_state_dict(checkpoint['model_state_dicts'][ae_index])
                    
                    if load_optimizer:
                        logging.info(f"Loading previous optimizer and scheduler states for autoencoder {ae_index}")
                        autoencoder.optimizers[ae_index].load_state_dict(checkpoint['optimizer_state_dicts'][ae_index])
                        autoencoder.schedulers[ae_index].load_state_dict(checkpoint['scheduler_state_dicts'][ae_index])
                    else:
                        logging.info(f"Resetting optimizer and scheduler for autoencoder {ae_index}")
                        autoencoder.optimizers[ae_index] = torch.optim.Adam(autoencoder.autoencoders[ae_index].parameters(), lr=0.0001)
                        autoencoder.schedulers[ae_index] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            autoencoder.optimizers[ae_index], mode='min', factor=0.4, patience=8, min_lr=1e-5
                        )
                else:
                    logging.info(f"No previous save found for autoencoder {ae_index}")
            # Move current autoencoder to GPU
            autoencoder.autoencoders[ae_index] = autoencoder.autoencoders[ae_index].to(device, dtype=dtype)
            # Move optimizer to GPU
            autoencoder.move_optimizer_to_device(autoencoder.optimizers[ae_index], device)
            
            # Use appropriate dataset for each autoencoder
            if ae_index == 2:
                dataset = FlattenedMultiAgentH5Dataset(h5_files_all_jammers, dtype=dtype)
            else:
                dataset = FlattenedMultiAgentH5Dataset(h5_files_low_jammers, dtype=dtype)
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            
            start_epoch = load_training_state(autoencoder, ae_index)
            
            best_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(start_epoch, num_epochs):
                # if interrupt_flag.value:
                #     print(f"Interrupt detected. Saving progress for autoencoder {ae_index}...")
                #     # save_training_state(autoencoder, ae_index, epoch)
                #     writer.flush()
                #     return

                total_loss = 0
                num_batches = 0
                
                with tqdm(dataloader, desc=f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs}") as t:
                    for batch_idx, batch in enumerate(t):
                        if interrupt_flag.value:
                            raise KeyboardInterrupt

                        if ae_index == 0:
                            layer_batch = batch['layer_0']
                        elif ae_index == 1:
                            layer_batch = torch.cat([batch['layer_1'], batch['layer_2']], dim=0)
                        else:  # ae_index == 2
                            layer_batch = batch['layer_3']
                        
                        loss = autoencoder.train_step(layer_batch, ae_index)
                        
                        if loss is not None:
                            total_loss += loss
                            num_batches += 1

                            # Update tqdm progress bar
                            t.set_postfix(loss=f"{loss:.4f}")

                            # Log at specified intervals
                            if batch_idx % log_interval == 0:
                                # logging.info(f'Autoencoder_{ae_index}/Batch_Loss', loss, epoch * len(dataloader) + batch_idx)
                                writer.add_scalar(f'Autoencoder_{ae_index}/Batch_Loss', loss, epoch * len(dataloader) + batch_idx)

                            # Adjust regularisation weights periodically
                            # if batch_idx % 100 == 0:
                            #     autoencoder.adjust_regularisation_weights()
                            #     autoencoder.adjust_l1_weight()

                        del layer_batch, loss
                        torch.cuda.empty_cache()

                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    print(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
                    logging.info(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
                    writer.add_scalar(f'Autoencoder_{ae_index}/Epoch_Loss', avg_loss, epoch)

                    autoencoder.schedulers[ae_index].step(avg_loss)

                    if abs(avg_loss - best_loss) <= delta:
                        epochs_no_improve += 1
                    else:
                        best_loss = min(avg_loss, best_loss)
                        epochs_no_improve = 0
                        autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_{ae_index}_best.pth"))

                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered for autoencoder {ae_index}. No improvement for {patience} epochs.")
                        logging.info(f"Early stopping triggered for autoencoder {ae_index}. No improvement for {patience} epochs.")
                        break
                else:
                    print(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} failed. All batches resulted in NaN loss.")
                    logging.warning(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} failed. All batches resulted in NaN loss.")

                save_training_state(autoencoder, ae_index, epoch)

                # if (epoch + 1) % 10 == 0:
                #     autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_{ae_index}_epoch_{epoch+1}.pth"))
                    # test_specific_autoencoder(autoencoder, H5_FOLDER, output_folder, autoencoder_index=ae_index, epoch=epoch+1)

                # torch.cuda.empty_cache()

            # After finishing all epochs for an autoencoder, move it back to CPU
            autoencoder.move_to_cpu(ae_index)

            print(f"Autoencoder {ae_index} training completed.")
            logging.info(f"Autoencoder {ae_index} training completed.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        log_error()
        raise
    finally:
        writer.close()
        autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
        logging.info(f"Autoencoder training completed and model saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")
    
def main():
    global interrupt_flag
    interrupt_flag = mp.Value('i', 0)
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    setup_logging()
    
    # Configuration ranges
    seed_range = range(1, 41) # 1-40 seed
    num_agents_range = range(10, 11) # 10 agents
    num_targets_range = range(90, 91) # 90 targets
    num_jammers_range_low = range(0, 1)  # 0 jammers
    num_jammers_range_high = range(85, 90)  # 85-89 jammers
    
    map_paths = ['city_image_1.npy', 'city_image_2.npy', 'city_image_3.npy']
    
    configs = [
        {'map_path': map_path, 'seed': seed, 'n_agents': num_agents, 'n_targets': num_targets, 'n_jammers': num_jammers}
        for map_path in map_paths
        for seed in seed_range
        for num_agents in num_agents_range
        for num_targets in num_targets_range
        for num_jammers in num_jammers_range_low
    ]
    
    # Additional configs for autoencoder 2
    configs_high_jammers = [
        {'map_path': map_path, 'seed': seed, 'n_agents': num_agents, 'n_targets': num_targets, 'n_jammers': num_jammers}
        for map_path in map_paths
        for seed in seed_range
        for num_agents in num_agents_range
        for num_targets in num_targets_range
        for num_jammers in num_jammers_range_high
    ]
    
    all_configs = configs + configs_high_jammers
    
    total_configs = len(all_configs)
    initial_completed = len(load_progress(all_configs))
    print(f"Initial completed configurations: {initial_completed}/{total_configs}")
    
    configs_to_process = []
    for config in all_configs:
        map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
        filename = f"data_m{map_name}_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
        filepath = os.path.join(H5_FOLDER, filename)
        if not os.path.exists(filepath) or not is_dataset_complete(filepath, STEPS_PER_EPISODE):
            configs_to_process.append((config, config_path, H5_FOLDER, STEPS_PER_EPISODE))
    
    print(f"Configurations to process: {len(configs_to_process)}")
    
    # Process configurations
    num_processes = max(1, mp.cpu_count() // 2 )
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        try:
            results = list(tqdm(
                pool.imap_unordered(process_config_wrapper, configs_to_process),
                total=len(configs_to_process),
                disable=interrupt_flag.value
            ))
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
        finally:
            pool.terminate()
            pool.join()
    
    completed_configs = load_progress(all_configs)
    print(f"Completed configurations: {len(completed_configs)}/{total_configs}")
    
    if len(completed_configs) >= total_configs:
        print("All configurations processed. Starting autoencoder training...")

        # Separate h5 files for regular and high-jammer configurations
        h5_files_low_jammers = [os.path.join(H5_FOLDER, filename) for filename in completed_configs if int(filename.split('_j')[1].split('_')[0]) < 5]
        h5_files_all_jammers = [os.path.join(H5_FOLDER, filename) for filename in completed_configs]
        
        with h5py.File(h5_files_low_jammers[0], 'r') as f:
            first_step = f['data']['0']
            first_agent = first_step[list(first_step.keys())[0]]
            # input_shape = first_agent['full_state'].shape

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = EnvironmentAutoencoder(device)

        # Train autoencoders
        train_autoencoder(autoencoder, h5_files_low_jammers, h5_files_all_jammers, batch_size=16)

        # Save the final autoencoder
        autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
        print(f"Final autoencoder saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")
    else:
        print(f"Not all configurations are complete. {len(completed_configs)}/{total_configs} configurations are ready.")
        print("Please run the script again to process the remaining configurations.")


def main_collect_test_data():
    # Constants
    H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
    TEST_FOLDER = os.path.join(H5_FOLDER, 'test_data')
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    STEPS_PER_EPISODE = 100
    NUM_TEST_CONFIGS = 5  # Number of test configurations to generate

    os.makedirs(TEST_FOLDER, exist_ok=True)
    setup_logging()

    # Load the original config
    original_config = load_config(CONFIG_PATH)

    # Generate test configurations
    test_configs = []
    map_paths = ['city_image_1.npy', 'city_image_2.npy', 'city_image_3.npy']
    
    for _ in range(NUM_TEST_CONFIGS):
        config = original_config.copy()
        config.update({
            'map_path': np.random.choice(map_paths),
            'seed': np.random.randint(1000, 2000),  # Use a different seed range
            'n_agents': np.random.randint(5, 15),
            'n_targets': np.random.randint(10, 100),
            'n_jammers': np.random.randint(0, 5)
        })
        test_configs.append(config)

    # Prepare configurations for processing
    configs_to_process = [
        (config, CONFIG_PATH, TEST_FOLDER, STEPS_PER_EPISODE) for config in test_configs
    ]

    # Process configurations
    num_processes = max(1, os.cpu_count() // 2)
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        try:
            results = list(tqdm(
                pool.imap_unordered(process_config_wrapper, configs_to_process),
                total=len(configs_to_process),
                desc="Collecting test data"
            ))
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
        finally:
            pool.terminate()
            pool.join()

    # Count successfully collected configurations
    completed_configs = load_progress(test_configs)
    print(f"Completed test configurations: {len(completed_configs)}/{NUM_TEST_CONFIGS}")
    print(f"Test data collection complete. New configurations saved in {TEST_FOLDER}")

    
if __name__ == "__main__":
    try:
        # main()
        main_collect_test_data()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
    except Exception as e:
        print(f"An error occurred: {e}")
        log_error()
    finally:
        print("Cleaning up...")
        cleanup_resources()
        print("All resources cleaned up")