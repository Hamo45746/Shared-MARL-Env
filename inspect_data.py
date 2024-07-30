import numpy as np

def load_and_inspect_npy(file_path):
    # Load the data from the .npy file
    data = np.load(file_path)
    
    # Print the shape of the data
    print(f"Data shape: {data.shape}")
    
    # Print the first few samples to inspect
    print("First few samples of the data:")
    np.set_printoptions(threshold=2000, suppress=True, precision=1, linewidth=2000)
    print(data[-5:])  # Change the number to inspect more or fewer samples

if __name__ == "__main__":
    # Provide the path to your .npy file
    file_paths = [
        'outputs/combined_data_map_view.npy',
        'outputs/combined_data_agent.npy',
        'outputs/combined_data_target.npy',
        'outputs/combined_data_jammer.npy'
    ]
    
    for file_path in file_paths:
        np.set_printoptions(threshold=2000, suppress=True, precision=1, linewidth=2000)
        print(f"\nInspecting file: {file_path}")
        load_and_inspect_npy(file_path)