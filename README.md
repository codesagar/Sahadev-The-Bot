# Sahadev-The-Bot
A context based question answering bot using seq2seq model with attention.



## Additional files to be downloaded before execution
file = glove.6B.100d.txt
target_dest = ~/Sahadev-The-Bot/model/data/


### Commands
wget http://nlp.stanford.edu/data/glove.6B.zip
unizp glove.6B.zip ~/Sahadev-The-Bot/model/data/




function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
gdrive_download 14S5neLSygdDVujh-aIPRMkcyhFyrgENg experiment.zip


unzip experiment.zip