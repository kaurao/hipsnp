# hispnp

The Forschungszentrum Jülich to o handle SNP data, especially from the UKB.

Check our full documentation here: https://juaml.github.io/hipsnp/main/index.html


## Licensing

hispnp is released under the AGPL v3 license:

hispnp, FZJuelich AML SNP data library.
Copyright (C) 2020, authors of julearn.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


## Citing
We still do not have a publication that you can use to cite hispnp in your
manuscript. 




# Example

Example with a mock dataset hosted on GIN

```Python
from hipsnp.hipsnp import genotype_from_datalad
from tempfile import mkdtemp

# temorary directory
workdir = mkdtemp()

# data source
source = 'https://gin.g-node.org/juaml/datalad-example-bgen'

# List of RSID that we want to obtain
rsids_of_interest = ['RSID_2', 'RSID_5', 'RSID_6', 'RSID_7']
chromosomes = ['1'] * len(rsids_of_interest)

genotype = genotype_from_datalad(rsids=rsids_of_interest,
                                 chromosomes=chromosomes,
                                 datalad_source=source,
                                 workdir=workdir,
                                 datadir=workdir)

genotype.metadata
	   REF	ALT	CHROM	POS	ID	FORMAT
RSID_2	A	G	01	2000	SNPID_2	GP
RSID_5	A	G	01	5000	SNPID_5	GP
RSID_6	A	G	01	6000	SNPID_6	GP
RSID_7	A	G	01	7000	SNPID_7	GP
```

# resources


## SNP databases

https://www.ncbi.nlm.nih.gov/snp/

http://www.ensembl.org/Homo_sapiens

https://varsome.com


## Tools required

https://www.well.ox.ac.uk/~gav/qctool/


## Info

https://eu.idtdna.com/pages/education/decoded/article/genotyping-terms-to-know

https://faculty.washington.edu/browning/intro-to-vcf.html

https://www.reneshbedre.com/blog/vcf-fields.html

https://www.garvan.org.au/research/kinghorn-centre-for-clinical-genomics/learn-about-genomics/for-gp/genetics-refresher-1/types-of-variants

