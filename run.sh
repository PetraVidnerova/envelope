for I in `seq 0 4`
do
   python test.py $I 0.1 0.9 mlp > /dev/null &
   python test.py $I 0.2 0.8 mlp > /dev/null &
   python test.py $I 0.3 0.7 mlp > /dev/null & 
done

echo "waiting"
wait
echo "ready steady go" 

DATA="data6"


for I in `seq 0 9`
do
    if test ! -f mlp_${DATA}_${I}_plain.txt
    then 
	python test_plain.py $I mlp > mlp_${DATA}_${I}_plain.txt  &
    fi
    if test ! -f mlp_${DATA}_${I}_0.1_0.9.txt
    then
	python test_plain.py $I mlp_${DATA}_${I}_y_0.1.npy mlp_${DATA}_${I}_y_0.9.npy mlp > mlp_${DATA}_${I}_0.1_0.9.txt  &
    fi
    if test ! -f mlp_${DATA}_${I}_0.2_0.8.txt
    then
    python test_plain.py $I mlp_${DATA}_${I}_y_0.2.npy mlp_${DATA}_${I}_y_0.8.npy mlp > mlp_${DATA}_${I}_0.2_0.8.txt  &
    fi
    if test ! -f mlp_${DATA}_${I}_0.3_0.7.txt
    then
	python test_plain.py $I mlp_${DATA}_${I}_y_0.3.npy mlp_${DATA}_${I}_y_0.7.npy mlp > mlp_${DATA}_${I}_0.3_0.7.txt   &
    fi
done    
