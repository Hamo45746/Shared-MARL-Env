import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import logging
import psutil
import time
import gc
import signal
import fcntl
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import traceback

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

logging.basicConfig(filename='autoencoder_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global flag to indicate interruption
interrupt_flag = mp.Value('i', 0)

class FlattenedMultiAgentH5Dataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.file_indices = []
        self.cumulative_lengths = [0]
        
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                length = sum(len(step) for step in f['data'].values())
            self.file_indices.extend([file_idx] * length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        with h5py.File(self.h5_files[file_idx], 'r') as f:
            data = f['data']
            cumulative_step_length = 0
            for step_idx, step_data in data.items():
                step_length = len(step_data)
                if local_idx < cumulative_step_length + step_length:
                    agent_idx = local_idx - cumulative_step_length
                    agent_id = list(step_data.keys())[agent_idx]
                    full_state = step_data[agent_id]['full_state'][()]
                    return {f'layer_{i}': torch.FloatTensor(full_state[i]) for i in range(full_state.shape[0])}
                cumulative_step_length += step_length
        
        raise IndexError(f"Index {idx} is out of bounds")

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def signal_handler(signum, frame):
    print("Interrupt received, stopping processes...")
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

def load_progress():
    progress_file = os.path.join(H5_FOLDER, H5_PROGRESS_FILE)
    lock_file = f"{progress_file}.lock"
    lock_fd = acquire_lock(lock_file)
    try:
        progress = set()
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                file_list = f.read().splitlines()
            
            for filename in file_list:
                filepath = os.path.join(H5_FOLDER, filename)
                if os.path.exists(filepath) and is_dataset_complete(filepath, STEPS_PER_EPISODE):
                    progress.add(filename)
                else:
                    print(f"Warning: File in progress list is missing or incomplete: {filename}")
        
        # Check for any completed files not in the progress file
        for filename in os.listdir(H5_FOLDER):
            if filename.endswith('.h5'):
                filepath = os.path.join(H5_FOLDER, filename)
                if is_dataset_complete(filepath, STEPS_PER_EPISODE) and filename not in progress:
                    progress.add(filename)
        
        # Update the progress file
        with open(progress_file, 'w') as f:
            for filename in progress:
                f.write(f"{filename}\n")
        
        return progress
    finally:
        release_lock(lock_fd)

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

def collect_data_for_config(config, config_path, steps_per_episode, h5_folder):
    env = Environment(config_path)
    env.config.update(config)
    env.num_agents = config['n_agents']
    env.reset()

    # Update filename to include map information
    map_name = os.path.splitext(os.path.basename(config['map_path']))[0]
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
            progress = load_progress()
            progress.add(os.path.basename(filepath))
            save_progress(progress)
            return filepath
        else:
            return None
    except Exception as e:
        logging.error(f"Error processing config {config}: {str(e)}")
        return None

def save_training_state(autoencoder, layer, epoch):
    state = {
        'model_state_dicts': [ae.state_dict() for ae in autoencoder.autoencoders],
        'optimizer_state_dicts': [opt.state_dict() for opt in autoencoder.optimizers],
        'layer': layer,
        'epoch': epoch
    }
    torch.save(state, os.path.join(H5_FOLDER, f"training_state_layer_{layer}.pth"))

def load_training_state(autoencoder, layer):
    state_path = os.path.join(H5_FOLDER, f"training_state_layer_{layer}.pth")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=autoencoder.device)
        autoencoder.autoencoders[layer].load_state_dict(state['model_state_dicts'][layer])
        autoencoder.optimizers[layer].load_state_dict(state['optimizer_state_dicts'][layer])
        return state['epoch']
    return 0

def visualise_autoencoder_progress(autoencoder, h5_folder, epoch, output_folder, ae_index):
    # Store the original device
    original_device = next(autoencoder.autoencoders[ae_index].parameters()).device
    
    try:
        # Move autoencoder to CPU for visualization
        autoencoder.autoencoders[ae_index] = autoencoder.autoencoders[ae_index].cpu()
        
        # Find a suitable H5 file (with high number of targets and agents)
        suitable_file = None
        for filename in os.listdir(h5_folder):
            if filename.endswith('.h5') and 'a14' in filename and 't42' in filename:
                suitable_file = os.path.join(h5_folder, filename)
                break
        
        if not suitable_file:
            logging.warning("No suitable H5 file found for visualization.")
            return

        # Load data from the H5 file
        with h5py.File(suitable_file, 'r') as f:
            first_step = f['data']['0']
            first_agent = first_step[list(first_step.keys())[0]]
            full_state = first_agent['full_state'][()]

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Autoencoder {ae_index} - Epoch {epoch}')

        # Determine which layer(s) to visualize based on ae_index
        if ae_index == 0:
            layers_to_visualize = [0]
        elif ae_index == 1:
            layers_to_visualize = [1, 2]
        else:  # ae_index == 2
            layers_to_visualize = [3]

        for layer in layers_to_visualize:
            # Input
            im_input = axes[0].imshow(full_state[layer], cmap='viridis')
            axes[0].set_title(f'Input (Layer {layer})')
            plt.colorbar(im_input, ax=axes[0], fraction=0.046, pad=0.04)

            # Output
            with torch.no_grad():
                input_tensor = torch.FloatTensor(full_state[layer]).unsqueeze(0).unsqueeze(0)
                encoded = autoencoder.autoencoders[ae_index].encode(input_tensor)
                decoded = autoencoder.autoencoders[ae_index].decoder(encoded)
                decoded = decoded.numpy().squeeze()

            im_output = axes[1].imshow(decoded, cmap='viridis')
            axes[1].set_title(f'Output (Layer {layer})')
            plt.colorbar(im_output, ax=axes[1], fraction=0.046, pad=0.04)

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'autoencoder_{ae_index}_layer_{layer}_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        logging.info(f"Visualisations for autoencoder {ae_index}, epoch {epoch} saved in {output_folder}")
    
    except Exception as e:
        logging.error(f"Error during visualization for autoencoder {ae_index}, epoch {epoch}: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Move autoencoder back to the original device
        autoencoder.autoencoders[ae_index] = autoencoder.autoencoders[ae_index].to(original_device)

def train_autoencoder(autoencoder, h5_files_low_jammers, h5_files_all_jammers, num_epochs=100, batch_size=32, patience=5, delta=0.0001):
    output_folder = os.path.join(H5_FOLDER, 'training_visualisations')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(H5_FOLDER, 'tensorboard_logs'))

    try:
        for ae_index in range(3):  # We have 3 autoencoders
            print(f"Training autoencoder {ae_index}")
            
            # Use appropriate dataset for each autoencoder
            if ae_index == 2:
                dataset = FlattenedMultiAgentH5Dataset(h5_files_all_jammers)
            else:
                dataset = FlattenedMultiAgentH5Dataset(h5_files_low_jammers)
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            
            start_epoch = load_training_state(autoencoder, ae_index)
            
            # Early stopping variables
            best_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(start_epoch, num_epochs):
                if interrupt_flag.value:
                    print(f"Interrupt detected. Saving progress for autoencoder {ae_index}...")
                    save_training_state(autoencoder, ae_index, epoch)
                    writer.flush()
                    return

                try:
                    total_loss = 0
                    num_batches = 0
                    nan_batches = 0
                    
                    for batch in tqdm(dataloader, desc=f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs}"):
                        if interrupt_flag.value:
                            raise KeyboardInterrupt

                        if ae_index == 0:
                            layer_batch = {f'layer_0': batch[f'layer_0']}
                        elif ae_index == 1:
                            layer_batch = {
                                f'layer_1': torch.cat([batch[f'layer_1'], batch[f'layer_2']], dim=0)
                            }
                        else:  # ae_index == 2
                            layer_batch = {f'layer_3': batch[f'layer_3']}
                        
                        loss, gradient_norm, weight_update_norm = autoencoder.train_step(layer_batch, ae_index)
                        
                        if loss is None:  # nan loss detected
                            nan_batches += 1
                            continue
                        
                        total_loss += loss
                        num_batches += 1

                        # Log additional metrics
                        writer.add_scalar(f'Autoencoder_{ae_index}/Batch_Loss', loss, epoch * len(dataloader) + num_batches)
                        writer.add_scalar(f'Autoencoder_{ae_index}/Gradient_Norm', gradient_norm, epoch * len(dataloader) + num_batches)
                        writer.add_scalar(f'Autoencoder_{ae_index}/Weight_Update_Norm', weight_update_norm, epoch * len(dataloader) + num_batches)

                        # Free up memory
                        del layer_batch
                        torch.cuda.empty_cache()

                    if num_batches > 0:
                        avg_loss = total_loss / num_batches
                        print(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
                        print(f"NaN losses in {nan_batches} out of {num_batches + nan_batches} batches")
                        logging.info(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
                        logging.info(f"NaN losses in {nan_batches} out of {num_batches + nan_batches} batches")

                        # Log epoch-level metrics
                        writer.add_scalar(f'Autoencoder_{ae_index}/Epoch_Loss', avg_loss, epoch)
                        writer.add_scalar(f'Autoencoder_{ae_index}/NaN_Batches', nan_batches, epoch)

                        # Check for improvement
                        if avg_loss < best_loss - delta:
                            best_loss = avg_loss
                            epochs_no_improve = 0
                            # Save the best model
                            autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_{ae_index}_best.pth"))
                        else:
                            epochs_no_improve += 1

                        # Check early stopping condition
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered for autoencoder {ae_index}")
                            logging.info(f"Early stopping triggered for autoencoder {ae_index}")
                            break

                    else:
                        print(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} failed. All batches resulted in NaN loss.")
                        logging.warning(f"Autoencoder {ae_index}, Epoch {epoch+1}/{num_epochs} failed. All batches resulted in NaN loss.")

                    save_training_state(autoencoder, ae_index, epoch + 1)

                    if (epoch + 1) % 10 == 0:
                        autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_{ae_index}_epoch_{epoch+1}.pth"))
                        # Generate and save visualizations
                        visualise_autoencoder_progress(autoencoder, H5_FOLDER, epoch + 1, output_folder, ae_index)

                    mem_percent = psutil.virtual_memory().percent
                    logging.info(f"Memory usage: {mem_percent}%")

                    if mem_percent > 90:
                        logging.warning("High memory usage detected. Pausing for 60 seconds.")
                        time.sleep(60)

                except KeyboardInterrupt:
                    print(f"Interrupt detected. Saving progress for autoencoder {ae_index}...")
                    save_training_state(autoencoder, ae_index, epoch)
                    writer.flush()
                    return

                except Exception as e:
                    logging.error(f"Error during training autoencoder {ae_index}, epoch {epoch+1}: {str(e)}")
                    raise

            print(f"Autoencoder {ae_index} training completed.")
            logging.info(f"Autoencoder {ae_index} training completed.")

            # After finishing all epochs for an autoencoder, move it back to CPU
            autoencoder.move_to_cpu(ae_index)

    finally:
        writer.close()
        autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
        logging.info(f"Autoencoder training completed and model saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")
    
    
def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    
    # Configuration ranges
    seed_range = range(1, 4)
    num_agents_range = range(12, 15)
    num_targets_range = range(42, 45)
    num_jammers_range_low = range(0, 4)  # 0-3 jammers
    num_jammers_range_high = range(85, 91)  # 85-90 jammers
    
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
    initial_completed = len(load_progress())
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
    num_processes = max(1, mp.cpu_count() - 1)
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
    
    completed_configs = load_progress()
    print(f"Final completed configurations: {len(completed_configs)}/{total_configs}")
    
    if len(completed_configs) == total_configs:
        print("All configurations processed. Starting autoencoder training...")

        # Separate h5 files for regular and high-jammer configurations
        h5_files_low_jammers = [os.path.join(H5_FOLDER, filename) for filename in completed_configs if int(filename.split('_j')[1].split('_')[0]) < 5]
        h5_files_all_jammers = [os.path.join(H5_FOLDER, filename) for filename in completed_configs]
        
        with h5py.File(h5_files_low_jammers[0], 'r') as f:
            first_step = f['data']['0']
            first_agent = first_step[list(first_step.keys())[0]]
            input_shape = first_agent['full_state'].shape

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = EnvironmentAutoencoder(input_shape, device)

        # Train autoencoders
        train_autoencoder(autoencoder, h5_files_low_jammers, h5_files_all_jammers)

        # Save the final autoencoder
        autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
        print(f"Final autoencoder saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")
    else:
        print(f"Not all configurations are complete. {len(completed_configs)}/{total_configs} configurations are ready.")
        print("Please run the script again to process the remaining configurations.")

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        cleanup_resources()
        print("All resources cleaned up")