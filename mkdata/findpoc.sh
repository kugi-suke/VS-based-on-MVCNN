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
  echo $line
  ./ghecom -M P -rl 4.0 -ipdb ./$I_VALUE/${line}.pdb -opocpdb ./$O_VALUE/${line}_pocks.pdb
done << FILE
$DATA
FILE
