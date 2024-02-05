
def reset_overlays():
    """
    Clears view and completely remove visualization. All files opened in FSLeyes are closed.
    The view (along with any color map) is reset to the regular ortho panel.
    """
    l = frame.overlayList
    while(len(l)>0):
        del l[0]
    frame.removeViewPanel(frame.viewPanels[0])
    # Put back an ortho panel in our viz for future displays
    frame.addViewPanel(OrthoPanel)
    
def mkdir_no_exist(path):
    if not op.isdir(path):
        os.makedirs(path)
        
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def direct_file_download_open_neuro(file_list, file_types, dataset_id, dataset_version, save_dirs):
    # https://openneuro.org/crn/datasets/ds004226/snapshots/1.0.0/files/sub-001:sub-001_scans.tsv
    for i, n in enumerate(file_list):
        subject = n.split('_')[0]
        download_link = 'https://openneuro.org/crn/datasets/{}/snapshots/{}/files/{}:{}:{}'.format(dataset_id, dataset_version, subject, file_types[i],n)
        print('Attempting download from ', download_link)
        download_url(download_link, op.join(save_dirs[i], n))
        print('Ok')
        
def get_json_from_file(fname):
    f = open(fname)
    data = json.load(f)
    f.close()
    return data
import openneuro
from mne.datasets import sample
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
#bids_root

def open_subject_dataset(dataset_id, subject, download = False):
    sub = f'sub-{subject}'
    sample_path = "dataset"
    mkdir_no_exist(sample_path)
    bids_root = op.join(os.path.abspath(""),sample_path, dataset_id)
    mkdir_no_exist(bids_root)

    #Derivatives folder to store masks and preprocessing garbage
    mkdir_no_exist(op.join(bids_root, 'derivatives'))
    preproc_root = op.join(bids_root, 'derivatives','preprocessed_data')
    mkdir_no_exist(preproc_root)
    mkdir_no_exist(op.join(preproc_root, sub))
    mkdir_no_exist(op.join(preproc_root, sub, 'anat'))
    mkdir_no_exist(op.join(preproc_root, sub, 'func'))
    mkdir_no_exist(op.join(preproc_root, sub, 'fmap'))
    
    if (download):
        os.system(f"rm {bids_root}/dataset_description.json")
        os.system("""openneuro-py download --dataset {} --include sub-{}/anat/* 
             --include sub-{}/func/sub-{}_task-flanker_run-1_bold.nii.gz 
             --include sub-{}/func/sub-{}_task-flanker_run-2_bold.nii.gz 
              --target_dir {}""".format(dataset_id, subject, subject, subject, subject, subject, bids_root).replace("\n", " "))
    
    print(f"Loaded {dataset_id} for {sub}")
    return bids_root, preproc_root

def perform_default_moco(subject):
    runs=2
    sub = f'sub-{subject}'
    for r in range(runs):
        path_original_data = os.path.join(bids_root, sub, 'func', f'{sub}_task-flanker_run-{r+1}_bold')
        path_moco_data = os.path.join(preproc_root, sub, 'func', f'{sub}_task-flanker_run-{r+1}_bold_moco')
        mcflirt(infile=path_original_data,o=path_moco_data, plots=True, report=True, dof=6, mats=True)

def load_mot_params_fsl_6_dof(subject):
    path=op.join(preproc_root, f'sub-{subject}', 'func', f'sub-{subject}_task-flanker_run-{flanker_run}_bold_moco.par')
    return pd.read_csv(path, sep='  ', header=None, 
            engine='python', names=['Rotation x', 'Rotation y', 'Rotation z','Translation x', 'Translation y', 'Translation z'])

def visualize_motparams(data):
    fig, ax = plt.subplots(3,2, figsize = (9, 8))
    for cdx, c in enumerate(data.columns.tolist()):
        ax[cdx%3, int(cdx/3)].plot(data[c], color = ["royalblue", "tomato"][int(cdx/3)])
        ax[cdx%3, int(cdx/3)].set_xlabel("Volumes")
        ax[cdx%3,int(cdx/3)].set_ylabel("Motion (mm)")
        ax[cdx%3,int(cdx/3)].set_title(c)
    plt.tight_layout()
    fig.suptitle(f"Flanker Run {flanker_run}", y = 1.02) 
    plt.show()

def compute_FD_power(mot_params):
    framewise_diff = mot_params.diff().iloc[1:]

    rot_params = framewise_diff[['Rotation x', 'Rotation y', 'Rotation z']]
    # Estimating displacement on a 50mm radius sphere
    # To know this one, we can remember the definition of the radian!
    # Indeed, let the radian be theta, the arc length be s and the radius be r.
    # Then theta = s / r
    # We want to determine here s, for a sphere of 50mm radius and knowing theta. Easy enough!
    
    # Another way to think about it is through the line integral along the circle.
    # Integrating from 0 to theta with radius 50 will give you, unsurprisingly, r0 theta.
    converted_rots = rot_params*50
    trans_params = framewise_diff[['Translation x', 'Translation y', 'Translation z']]
    fd = converted_rots.abs().sum(axis=1) + trans_params.abs().sum(axis=1)
    return fd

def visualise_fd(fd, subject):
    threshold = np.quantile(fd,0.75) + 1.5*(np.quantile(fd,0.75) - np.quantile(fd,0.25))
    plt.plot(list(range(1, fd.size+1)), fd)
    plt.title(f"Framewise Displacement (FD) for all the volumes of Flanker Run {flanker_run} for sub-{subject}")
    plt.xlabel('Volume')
    plt.ylabel('FD displacement (mm)')
    plt.hlines(threshold, 0, 150,colors='black', linestyles='dashed', label='FD threshold')
    plt.legend()
    plt.show()

# Compute RMS Movement from Transformation Parameters:
def calculate_rms_movement(subject):
    par_file=op.join(preproc_root, f'sub-{subject}', 'func', f'sub-{subject}_task-flanker_run-{flanker_run}_bold_moco.par')
    # Load motion parameters from the .par file
    motion_params = np.genfromtxt(par_file)
    # Extract translation parameters (mm)
    translation_params = motion_params[:, 3:6]
    # Calculate the RMS movement for each time point
    rms_movement = np.sqrt(np.sum(translation_params**2, axis=1))
    average_rms_movement = np.mean(rms_movement)
    return average_rms_movement

def visualize_motparams_oneplot(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['r', 'g', 'b']
    
    for i, coord in enumerate(['x', 'y', 'z']):
        rotation_col = f'Rotation {coord}'
        translation_col = f'Translation {coord}'

        ax.plot(data.index, data[rotation_col], label=f'Rotation {coord}', linestyle='--', color=colors[i])
        ax.plot(data.index, data[translation_col], label=f'Translation {coord}', color=colors[i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Values (mm)')
    ax.set_title('Rotation and Translation Data')
    ax.legend()
    plt.grid()
    plt.show()

def calculate_rms_movement_path(path):
    par_file=op.join(preproc_root, f'sub-{subject}', 'func', path)
    # Load motion parameters from the .par file
    motion_params = np.genfromtxt(par_file)
    # Extract translation parameters (mm)
    translation_params = motion_params[:, 3:6]
    # Calculate the RMS movement for each time point
    rms_movement = np.sqrt(np.sum(translation_params**2, axis=1))
    average_rms_movement = np.mean(rms_movement)
    return average_rms_movement
    
def load_mot_params_fsl_6_dof_path(path):
    path=op.join(preproc_root, f'sub-{subject}', 'func', path)
    return pd.read_csv(path, sep='  ', header=None, 
            engine='python', names=['Rotation x', 'Rotation y', 'Rotation z','Translation x', 'Translation y', 'Translation z'])
