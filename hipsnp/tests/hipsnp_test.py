import pytest
import os
import hipsnp as hps
import json
import pandas as pd
import datalad.api as dl
from pathlib import Path, PosixPath

mock_rsid = 'rs699'

def validateJSON(jsonData):
    """attemts to open a JSON var"""
    try:
        json_object = json.loads(jsonData)
    except ValueError as e:
        return False
    return True


def test_JSON_format_ensembl_human_rsid():
    """test output is in JSON format"""
    mock_rsid = 'rs699'
    outJSON = hps.ensembl_human_rsid(mock_rsid)
    assert validateJSON(outJSON)


def validatePANDAStype(pdData):
    """Data is a pandas DataFrame with fields 'chromosomes' and 'rsids' of type str"""
    assert isinstance(pdData, pd.core.frame.DataFrame)

def test_rsid2chromosome_has_pandas_format():
    mock_rsid = 'rs699'
    outPANDAS = hps.rsid2chromosome(mock_rsid)
    validatePANDAStype(outPANDAS)

def valiadtePANDAS_has_RSDIandCROMOSOM(pdData, refColFields):
    outFields = [field for field in pdData.columns]
    assert refColFields.sort() == outFields.sort()

def test_rsid2chromosome_has_RSIDandCROMOSOM():
    mock_rsid = 'rs699'
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)


def test_rsid2chromosome_has_list_of_RSIDandCROMOSOM():
    mock_rsid = ['rs699', 'rs698', 'rs101']
    refColFields = ['rsids','chromosomes']
    outPANDAS = hps.rsid2chromosome(mock_rsid)

    valiadtePANDAS_has_RSDIandCROMOSOM(outPANDAS, refColFields)

def removeDATALADdataset(dlObject):
    dlObject.remove()

def test_datalad_get_chromosome_return_DataladType():
    """Read example data from GIN, assert returnst datalad object"""
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    
    _,dataladObject,_ = hps.datalad_get_chromosome(c=c,source=source)
    assert type(dataladObject) == dl.Dataset
    removeDATALADdataset(dataladObject)

def validateNameObtainedFiles(dataLget):
    """files obtined with DataLad are the exnple files"""
    filenames = [os.path.basename(ind['path']) for ind in dataLget if ind['type'] == 'file']
    sameFiles = 'example_c1_v0.bgen' and 'example_c1_v0.sample' in filenames
    return sameFiles

def test_datalad_get_chromosome_dataland_get():
    """ dataland.get()_returns_expected_data structure """
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN

    _,dataladObject,dataGet = hps.datalad_get_chromosome(c=c,source=source)
    assert validateNameObtainedFiles(dataGet)
    removeDATALADdataset(dataladObject)

def test_datalad_get_chromosome_file_paths():
    """ returns the path to the files installed by DL """
    c = '1'
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    filesRef = [Path('/home/oportoles/Documents/MyCode/hipsnp/test_data/imputation/example_c1_v0.bgen'),
                Path('/home/oportoles/Documents/MyCode/hipsnp/test_data/imputation/example_c1_v0.sample')]
    # outdir = '/home/oportoles/Documents/MyCode/hipsnp/test_data'
    outdir = Path('/home/oportoles/Documents/MyCode/hipsnp/test_data')
    files,dataladObject,_ = hps.datalad_get_chromosome(c=c,source=source,path=outdir)
    assert sorted(files) == sorted(filesRef)
    removeDATALADdataset(dataladObject)


def test_get_chromosome_from_RSIDs():
    """get chromosom from user given RSID, chek return datalad dataset and paths to data"""
    datalad_source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    rsids = 'rs101'
    ### copied from rsid2vcf
    ch_rs = hps.rsid2chromosome(rsids, chromosomes=None)
    chromosomes = ch_rs['chromosomes'].tolist()
    uchromosomes = pd.unique(chromosomes)
    print('chromosomes needed: ' + str(uchromosomes) + '\n')
    files, ds, getout = [None]*len(uchromosomes), [None]*len(uchromosomes), [None]*len(uchromosomes)
    for c in range(len(uchromosomes)):
        ch = uchromosomes[c]
        files[c], ds[c], getout[c] = hps.datalad_get_chromosome(ch, source=datalad_source, path=None)
    ###
    filesOK = all([f == str or f == PosixPath for f in files])
    dsOK = all([type(d) == dl.Dataset for d in ds])
    assert filesOK and dsOK

# @pytest.mark.parametrize("qctool",
#                         [('/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'),
#                          ('/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/'),
#                          ('qctool')])
# def test_rsid2vcf_qctool(qctool):
def test_rsid2vcf():
    """ finds and uses qctool"""
    source = 'git@gin.g-node.org:/juaml/datalad-example-bgen' # exmaple data on GIN
    outdir = '/home/oportoles/Documents/MyCode/hipsnp/test_data'
    rsids = ['rs101']
    chromosomes = ['1']
    qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64/qctool'

    # qctool = '/home/oportoles/Apps/qctool_v2.0.6-Ubuntu16.04-x86_64'
    
    # returns: a pandas dataframe with rsid-chromosome pairs
    #  and the vcf files are created in the outdir
    # ch_rs: pandas DataFrame 'chromosomes' 'rsids'
    # files: 
    # ds: datalad DS
    ch_rs, _, dataL = hps.rsid2vcf(rsids, outdir,        
                                    datalad_source=source,
                                    qctool=qctool,
                                    datalad_drop=True,
                                    datalad_drop_if_got=True,
                                    datalad_dir=None,
                                    force=False,
                                    chromosomes=chromosomes,
                                    chromosomes_use=None)
    
    assert type(dataL) == dl.Dataset 
    removeDATALADdataset(dataL)


