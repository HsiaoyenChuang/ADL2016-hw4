import subprocess
import stat
import os
str = 'BLEU = 0.95, 4.4 / 0.9 / 0.5 / 0.4(BP=1.000, ratio=1.177, hyp_len=35759339, ref_len=30376256)'

def get_bleu(filename,targ):
    #data/data/valid.en < data/data/valid.es > result.txt
    #BLEU = 0.95, 4.4 / 0.9 / 0.5 / 0.4(BP=1.000, ratio=1.177, hyp_len=35759339, ref_len=30376256)
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/multi-bleu.perl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval,filename , ' < ',targ],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    out = None


    for line in stdout.split(','):
        if 'BLEU =' in line:
            return  line.replace('BLEU =','')
    if out == None:
        return '0'

if __name__== '__main__':
    bleu = get_bleu('data/' + 'valid_result.txt', 'data/' + 'data/valid.es')