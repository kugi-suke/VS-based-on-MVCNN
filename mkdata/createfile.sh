while getopts l:i:o: OPT; do
  case $OPT in
    l) L_VALUE=$OPTARG;;  ## input proteinlist
    i) I_VALUE=$OPTARG;;  ## input directory
    o) O_VALUE=$OPTARG;;  ## output directory
  esac
done

DATA=`cat ./$L_VALUE`

while read line
do
  cat ./$I_VALUE/${line}_pocks.pdb | grep -v "REMARK" > testtmp.pdb
  sed -n "/TER/q;p" ./testtmp.pdb > ./$O_VALUE/${line}_pock.pdb
  rm testtmp.pdb
done << FILE
$DATA
FILE
