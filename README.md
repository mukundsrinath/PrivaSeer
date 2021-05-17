# Privacy at Scale: Introducing the PrivaSeer Corpus of Web Privacy Policies

This repository contains the PrivaSeer Corpus described in the paper [Privacy at Scale: Introducing the PrivaSeer Corpus of Web Privacy Policies](https://arxiv.org/abs/2004.11131). 

The PrivaSeer Corpus contains 1.4 million privacy policies in the form of HTML files. The HTML files have been distributed between 100 folders. Each folder has ~10,000 HTML files. The HTML files are named based on the hash value of the URL of the privacy policy web page. Each file name has the format hash_value.html. 

The metadata for each file is provided in the file named ‘metadata.’ The metadata file contains the following data regarding each file:

**hash**: The hash value of the URL of the privacy policy web page as well as the html file name <br />
**timestamp**: The date when the file was crawled in the format dd/mm/yyyy <br />
**url**: Privacy policy web page URL <br />
**folder_path**: The name of the folder in which the file is located. <br />
**probability**: The probability that the file is a privacy policy. Only those files which have a probability greater than 0.5 have been included in the corpus. Please refer to the paper for more details.

The corpus has been made available for research, teaching, and scholarship purposes only, under a CC BY-NC-SA license. Please contact us for any requests regarding commercial use. 

If you use this dataset as part of a publication, you must cite the following paper:
Srinath, Mukund, Shomir Wilson, and C. Lee Giles. "Privacy at scale: Introducing the privaseer corpus of web privacy policies." arXiv preprint arXiv:2004.11131 (2020).
