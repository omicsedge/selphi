#!/usr/bin/env python
# selphi-imputation 1.5.3

import os, subprocess, dxpy, glob


SELPHI_DIR = "/home/dnanexus/selphi"
PYTHON = "python3.11"


def setup_environment(need_xsqueezeit=False):
    """Install Python 3.11 and selphi dependencies."""
    subprocess.check_call(
        "add-apt-repository -y ppa:deadsnakes/ppa"
        " && apt-get update"
        " && apt-get install -y python3.11 python3.11-dev python3.11-venv",
        shell=True,
    )

    subprocess.check_call(
        f"{PYTHON} -m ensurepip --upgrade"
        f" && {PYTHON} -m pip install --upgrade pip",
        shell=True,
    )

    subprocess.check_call(
        f"{PYTHON} -m pip install -r {SELPHI_DIR}/requirements.txt",
        shell=True,
    )

    if need_xsqueezeit:
        _install_xsqueezeit()


def _install_xsqueezeit():
    """Compile xSqueezeIt from source (only needed for --ref_source_xsi)."""
    subprocess.check_call(
        "apt-get install -y git cmake libcurl4-openssl-dev libhts-dev"
        " && git clone https://github.com/rwk-unil/xSqueezeIt.git /tmp/xsqueezeit"
        " && cd /tmp/xsqueezeit && mkdir build && cd build"
        " && cmake .. && make -j$(nproc)"
        " && cp /tmp/xsqueezeit/build/xsqueezeit /usr/local/bin/",
        shell=True,
    )


def get_ass_file_from_fileid(object_dnanexus, filename_extension, sep):
    names = object_dnanexus.name
    index_query = dxpy.find_data_objects(
        classname="file", name=f"{names}{sep}{filename_extension}"
    )
    index_object = next(index_query, None)

    if index_object is not None:
        id_index = index_object["id"]
        return dxpy.DXFile(id_index)
    else:
        print(f"No file found with name {names}{sep}{filename_extension}")

    return None


@dxpy.entry_point("main")
def main(
    refpanel,
    cores,
    target=None,
    ref_source_vcf=None,
    prepare_reference=None,
    map=None,
    outvcf=None,
    match_length=None,
    ref_source_xsi=None,
    tmp_path=None,
    est_ne=None,
    chunk_size=None,
    no_core_reduction=None,
):

    project_id = dxpy.PROJECT_CONTEXT_ID

    # Install Python 3.11 + dependencies (xSqueezeIt only if needed)
    setup_environment(need_xsqueezeit=(ref_source_xsi is not None))

    if target is not None:
        target = dxpy.DXFile(target)
    if ref_source_vcf is not None:
        ref_source_vcf = dxpy.DXFile(ref_source_vcf)
    if map is not None:
        map = dxpy.DXFile(map)
    if ref_source_xsi is not None:
        ref_source_xsi = dxpy.DXFile(ref_source_xsi)

    if target is not None:
        dxpy.download_dxfile(target.get_id(), "/target")

    if ref_source_vcf is not None:
        dxpy.download_dxfile(ref_source_vcf.get_id(), "/ref_source_vcf")
        ref_source_vcf_tbi = get_ass_file_from_fileid(ref_source_vcf, "tbi", ".")
        if ref_source_vcf_tbi is not None:
            dxpy.download_dxfile(ref_source_vcf_tbi.get_id(), "/ref_source_vcf.tbi")

    if map is not None:
        dxpy.download_dxfile(map.get_id(), "/map")

    if ref_source_xsi is not None:
        dxpy.download_dxfile(ref_source_xsi.get_id(), "/ref_source_xsi")
        ref_source_xsi_bcf = get_ass_file_from_fileid(ref_source_xsi, "bcf", "_var.")
        ref_source_xsi_csi = get_ass_file_from_fileid(
            ref_source_xsi, "csi", "_var.bcf."
        )
        dxpy.download_dxfile(ref_source_xsi_bcf.get_id(), "/ref_source_xsi_var.bcf")
        dxpy.download_dxfile(ref_source_xsi_csi.get_id(), "/ref_source_xsi_var.bcf.csi")

    subprocess.check_call(["mkdir", "-p", "/reference"])
    subprocess.check_call(["mkdir", "-p", "/output"])

    refpanel_prefix_name = os.path.basename(refpanel)
    output_folder = os.path.dirname(refpanel)

    # Build optional args
    extra_args = ""
    if match_length is not None:
        extra_args += f" --match_length {match_length}"
    if est_ne is not None:
        extra_args += f" --est_ne {est_ne}"
    if chunk_size is not None:
        extra_args += f" --chunk_size {chunk_size}"
    if no_core_reduction:
        extra_args += " --no_core_reduction"
    if tmp_path is not None:
        extra_args += f" --tmp_path {tmp_path}"

    if prepare_reference is not None and ref_source_vcf is not None:
        cmd = (
            f"{PYTHON} {SELPHI_DIR}/selphi.py"
            f" --prepare_reference"
            f" --ref_source_vcf /ref_source_vcf"
            f" --refpanel /reference/{refpanel_prefix_name}"
            f" --cores {cores}"
            f"{extra_args}"
        )
        subprocess.check_call(cmd, shell=True)

    if prepare_reference is not None and ref_source_xsi is not None:
        cmd = (
            f"{PYTHON} {SELPHI_DIR}/selphi.py"
            f" --prepare_reference"
            f" --ref_source_xsi /ref_source_xsi"
            f" --refpanel /reference/{refpanel_prefix_name}"
            f" --cores {cores}"
            f"{extra_args}"
        )
        subprocess.check_call(cmd, shell=True)

    if prepare_reference is not None:
        results = glob.glob("/reference/*")
        references_file = []
        for reference in results:
            file_obj = dxpy.upload_local_file(
                reference, project=project_id, folder=output_folder
            )
            references_file.append(file_obj.get_id())

        output = {}
        output["outvcf"] = [dxpy.dxlink(item) for item in references_file]

        return output

    else:

        output_name = os.path.basename(outvcf)
        output_folder = os.path.dirname(outvcf)

        dir_name = os.path.dirname(refpanel)
        for extension in [".srp", ".pbwt", ".sites", ".samples"]:
            complete_name = os.path.basename(refpanel) + extension
            file_obj = dxpy.find_one_data_object(
                classname="file",
                name=complete_name,
                folder=dir_name,
                project=project_id,
            )
            dxpy.download_dxfile(file_obj["id"], f"/reference/{complete_name}")
            if os.path.exists(f"/reference/{complete_name}"):
                print(f"{complete_name} file downloaded successfully.")
            else:
                print(f"{complete_name} file not found. Check the download process.")

        cmd = (
            f"{PYTHON} {SELPHI_DIR}/selphi.py"
            f" --target /target"
            f" --refpanel /reference/{refpanel_prefix_name}"
            f" --map /map"
            f" --outvcf /output/{output_name}"
            f" --cores {cores}"
            f"{extra_args}"
        )
        subprocess.check_call(cmd, shell=True)

    if os.path.exists(f"/output/{output_name}.vcf.gz"):
        print(f"{output_name}.vcf.gz output created successfully.")
    else:
        print(f"{output_name}.vcf.gz output not found. Error")

    results = glob.glob("/output/*")
    vcf_file = []
    for vcf in results:
        file_obj = dxpy.upload_local_file(vcf, project=project_id, folder=output_folder)
        vcf_file.append(file_obj.get_id())

    output = {}
    output["outvcf"] = [dxpy.dxlink(item) for item in vcf_file]

    return output


dxpy.run()
