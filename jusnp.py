import requests
import io
import os
import pandas as pd


def ensembl_human_rsid(rsid):
    """
    make a REST call to ensemble and return json info of a variant given a rsid
    """
    url = 'http://rest.ensembl.org/variation/human/' + rsid + '?content-type=application/json'
    response = requests.get(url)
    return response


def read_vcf(path):
    """
    taken shameless from: https://gist.github.com/dceoy/99d976a2c01e7f0ba1c813778f9db744
    Thanks Daichi Narushima
    """
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def vcf2genotype(vcf, th=0.9, snps=None, samples=None):
    """
    given a vcf file path or a pandas df from read_vcf returns genotypes
    """
    if isinstance(vcf ,str):
        vcf = read_vcf(vcf)
    elif isinstance(vcf, pd.DataFrame):
        pass
    else:
        print("don't know how to handle vcf")
        raise

    format = pd.unique(vcf['FORMAT'])
    if len(format) != 1 or format[0] != 'GP':
        print('I can only deal with the GP format')
        raise

    nsnp = vcf.shape[0]
    ncol = vcf.shape[1]
    if samples is None:
        samples = [vcf.columns[i] for i in range(9, ncol)]
    else:
        assert all(sam in list(vcf.columns) for sam in samples)


    if snps is None:
        snps = list(vcf['ID'])
    else:
        assert all(snp in list(vcf['ID']) for snp in snps)

    labels = pd.DataFrame(index=range(len(snps)), columns=range(len(samples)))
    labels.index = snps
    labels.columns = samples
    snps_index = [snps.index(snp) for snp in snps]
    for snp in snps_index:
        REF = vcf['REF'][snp]
        ALT = vcf['ALT'][snp]
        for sam in samples:
            GP = vcf[sam][snp]
            GP = [float(x) for x in GP.split(',')]
            f = lambda i: GP[i]
            GT = max(range(len(GP)), key=f)
            if GP[GT] >= th:
                if GT == 0:
                    labels[sam][snp] = REF + REF
                elif GT == 1:
                    labels[sam][snp] = REF + ALT
                else:
                    labels[sam][snp] = ALT + ALT

    return labels

