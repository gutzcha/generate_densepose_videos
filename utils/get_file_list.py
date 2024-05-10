import os
import glob

def save_list_as_text(file_list, save_path='file_list.txt'):
    with open(save_path, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)
def get_file_names(folder_name_list):
    if isinstance(folder_name_list, str):
        folder_name_list = [folder_name_list]
    
    all_file_names = []
    for foldername in folder_name_list:
        if not os.path.exists(foldername):
            Warning ("Folder does not exist: " + foldername)
            continue

        file_names = glob.glob(foldername + "/*.mp4")
        all_file_names += file_names
        
    return all_file_names


if __name__ == "__main__":
    folder_name_list = [
        '/lustre/fast/fast/ygoussha/miga_challenge/smg_split_files/train',
        '/lustre/fast/fast/ygoussha/miga_challenge/smg_split_files/validation']
    file_names = get_file_names(folder_name_list)
    # print(file_names[:10])
    print(len(file_names))

    save_list_as_text(file_names, 'all_data_list.txt')

    # exclue_name_list = [
    #     '/videos/mpi_data/2Itzik/MPIIGroupInteraction/densepose_train',
    #     '/videos/mpi_data/2Itzik/MPIIGroupInteraction/densepose_val',
    #     '/videos/mpi_data/2Itzik/MPIIGroupInteraction/densepose_test']
    # file_names = get_file_names(exclue_name_list)
    # # print(file_names[:10])
    # print(len(file_names))
    # save_list_as_text(file_names, 'exclude_list.txt')
