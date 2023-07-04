import h5py
import numpy as np

def compare_h5_files(file1, file2):
    with h5py.File(file1, 'r') as h5file1, h5py.File(file2, 'r') as h5file2:
        for dataset in h5file1.keys():
            if dataset not in h5file2:
                print(f"Dataset {dataset} not found in second file.")
                return False
            if dataset=='feats':
                if not np.array_equal(h5file1[dataset][:], h5file2[dataset][:]):
                    print(f"Data in dataset {dataset} not equal.")
                    return False

        # Check for datasets only present in the second file
        for dataset in h5file2.keys():
            if dataset not in h5file1:
                print(f"Dataset {dataset} not found in first file.")
                return False
        
        print("Files are equal.")
        return True

compare_h5_files('/mnt/ceph_vol/features/2020/debug1/h5_files/256px_resnet50_0mpp_8xdown_normal/20-001_K1_Movat_a.h5', '/mnt/ceph_vol/features/2020/debug1/h5_files/256px_resnet50_0mpp_8xdown_normal/20-001_K1_Movat.h5')
