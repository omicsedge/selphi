import numpy as np
import pandas as pd
import zarr
import pickle 
def vcf_to_zarr_target_full(
    vcf_file="./data/20.reference_panel.30x.hg38.HG00096.vcf.gz",
    output_name='./data/new_sample/target_full_array.zip',
    full_id_list_path="./data/new_sample/full_id_list_full.txt",
):
    """
    """
    target_full_array = pd.read_csv(vcf_file, sep="\t", comment="#", header=None)
    target_full_array.drop([0,1,3,4,5,6,7,8],axis=1,inplace=True)
    concat_target_full = pd.concat([
        target_full_array.applymap(lambda x: str(x).replace("/","|").split("|")[0]),
        target_full_array.applymap(lambda x: str(x).replace("/","|").split("|")[-1]),
    ],axis=1)
    concat_target_full
    concat_target_full.drop([2],axis=1,inplace=True)
    concat_target_full.columns = [0,1]

    full_id_list = list(target_full_array[2])
    with open(full_id_list_path, "wb") as fp:   #Pickling
        pickle.dump(full_id_list, fp)

    zarr.save(output_name, concat_target_full.to_numpy().astype(np.int8))


def vcf_to_target_chip(
    full_id_list,
    vcf_file="./data/HG00096_euro_15k_unphased.vcf.gz",
    output_name='./data/new_sample/target_chip_array.zip',
    chip_id_list_path="./data/new_sample/chip_id_list_full.txt",
):
    target_chip_array = pd.read_csv(vcf_file, sep="\t", header=None, comment="#")
    target_chip_array.drop([0,1,3,4,5,6,7,8],axis=1,inplace=True)
    target_chip_array = target_chip_array[target_chip_array[2].isin(full_id_list)]
    chip_id_list = list(target_chip_array[2])

    with open(chip_id_list_path, "wb") as fp:   #Pickling
        pickle.dump(chip_id_list, fp)

    target_chip_array.drop([2],1,inplace=True)
    concat_target = pd.concat(
        [target_chip_array.applymap(lambda x: str(x).replace("/","|").split("|")[0]),
        target_chip_array.applymap(lambda x: str(x).replace("/","|").split("|")[-1])],axis=1)
    zarr.save(output_name, concat_target.to_numpy().astype(np.int8))


def remove_sample_from_ref(
    indexes,
    ref_numpy_arr,
):
    """
    """
    ref_numpy_arr = np.delete(ref_numpy_arr, indexes, axis=1)

    return ref_numpy_arr


def get_sample_index(
    sample_name,
    samples_txt_path="/home/ec2-user/adriano/imputation/phase3/selphi-2/data/samples_HG00096_REMOVED.txt",
):
    """
    
    """
    df = pd.read_csv(samples_txt_path,header=None)
    indexes = df.index[df[0] == sample_name].tolist()
    indexes.append(indexes[0]+3201)

    return indexes